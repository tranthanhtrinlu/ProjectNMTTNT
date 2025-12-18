import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
from PIL import Image, ImageFile
import gc
import numpy as np
from collections import defaultdict
import time

# Fix cho ảnh lỗi
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from config import (
    TRAIN_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    VAL_BATCH_SIZE,
    NUM_EPOCHS,
    LR,
    DEVICE,
    EFFICIENTNET_MODEL_PATH,
    CLASSES,
    WEIGHT_DECAY,
    NUM_WORKERS,
    PREFETCH_FACTOR,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    USE_MIXED_PRECISION,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CLIP_NORM,
    EARLY_STOPPING_PATIENCE,
    MIN_DELTA,
    LR_SCHEDULER_PATIENCE,
    LR_SCHEDULER_FACTOR,
    MIN_LR,
    DROPOUT_RATE,
    DROP_PATH_RATE,
    LABEL_SMOOTHING,
    CLEAR_CACHE_EVERY_N_BATCHES,
    GC_COLLECT_EVERY_N_EPOCHS,
    RANDOM_ROTATION,
    COLOR_JITTER_BRIGHTNESS,
    COLOR_JITTER_CONTRAST,
    COLOR_JITTER_SATURATION,
    RANDOM_ERASING_PROB,
    SAVE_CHECKPOINT_EVERY,
)
from src.utils import compute_metrics


def robust_image_loader(path):
    """Custom image loader với error handling tốt hơn"""
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            img.load()  # Force load to catch errors early
            return img.convert('RGB')
    except Exception as e:
        print(f"\n⚠️  Cannot load {path}: {e}")
        return Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))


def get_dataloaders(use_subset=False, subset_size=1000):
    """
    Tạo dataloaders với data augmentation tối ưu
    
    Args:
        use_subset: Nếu True, chỉ dùng subset nhỏ để test nhanh
        subset_size: Kích thước subset
    """
    
    # Training transforms - Điều chỉnh theo IMAGE_SIZE
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(RANDOM_ROTATION),
        transforms.ColorJitter(
            brightness=COLOR_JITTER_BRIGHTNESS,
            contrast=COLOR_JITTER_CONTRAST,
            saturation=COLOR_JITTER_SATURATION,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet stats
        transforms.RandomErasing(p=RANDOM_ERASING_PROB, scale=(0.02, 0.15)),
    ])

    # Validation transforms - Không augmentation
    test_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load datasets
    print(f"\n📁 Loading datasets...")
    train_dataset = datasets.ImageFolder(
        root=str(TRAIN_DIR), 
        transform=train_tf,
        loader=robust_image_loader
    )
    test_dataset = datasets.ImageFolder(
        root=str(TEST_DIR), 
        transform=test_tf,
        loader=robust_image_loader
    )

    # Verify class mapping
    print(f"\n📊 Class Mapping:")
    print(f"  {train_dataset.class_to_idx}")
    
    expected_mapping = {'FAKE': 0, 'REAL': 1}
    if train_dataset.class_to_idx != expected_mapping:
        print(f"\n⚠️  WARNING: Class mapping mismatch!")
        raise ValueError("Training cancelled due to class mapping mismatch")
    else:
        print(f"  ✅ Class mapping is correct!")
   
    # Data distribution
    print(f"\n📊 Data Distribution:")
    print(f"  Train: {len(train_dataset):,} images")
    print(f"  Test:  {len(test_dataset):,} images")
    
    for idx, class_name in enumerate(train_dataset.classes):
        train_count = sum(1 for _, label in train_dataset.samples if label == idx)
        test_count = sum(1 for _, label in test_dataset.samples if label == idx)
        print(f"  {class_name}: {train_count:,} train / {test_count:,} test")

    # Subset for quick testing
    if use_subset:
        print(f"\n⚠️  Using subset of {subset_size} samples for testing")
        train_indices = np.random.choice(len(train_dataset), subset_size, replace=False)
        test_indices = np.random.choice(len(test_dataset), subset_size//4, replace=False)
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, test_loader


def build_model():
    """Build EfficientNet-B0 với config tối ưu"""
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=len(CLASSES),
        drop_rate=DROPOUT_RATE,
        drop_path_rate=DROP_PATH_RATE
    )
    return model.to(DEVICE)


def evaluate(model, data_loader, criterion, desc="Evaluating"):
    """Đánh giá model trên validation set"""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=desc, leave=False):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # Mixed precision inference
            with torch.amp.autocast('cuda', enabled=USE_MIXED_PRECISION):
                outputs = model(images)
                loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)[:, 1]

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / len(data_loader.dataset)
    metrics = compute_metrics(all_labels, all_probs)

    return avg_loss, metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """Lưu checkpoint với metadata đầy đủ"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'config': {
            'IMAGE_SIZE': IMAGE_SIZE,
            'BATCH_SIZE': BATCH_SIZE,
            'LR': LR,
        }
    }
    torch.save(checkpoint, filepath)


def train():
    """Main training function với tối ưu cho dataset lớn"""
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"\n🎮 GPU Information:")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        avail_mem = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"  Total Memory: {total_mem:.2f} GB")
        print(f"  Available Memory: {avail_mem:.2f} GB")
        
        # VRAM estimate
        vram_per_image = (IMAGE_SIZE * IMAGE_SIZE * 3 * 4) / 1024**3
        estimated_vram = vram_per_image * BATCH_SIZE * 3.5
        print(f"  Estimated Usage: ~{estimated_vram:.2f} GB")
        
        if estimated_vram > 3.8:
            print(f"\n  ⚠️  WARNING: High VRAM usage detected!")
            response = input(f"     Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Training cancelled.")
                return
   
    # Check existing model
    if EFFICIENTNET_MODEL_PATH.exists():
        print(f"\n⚠️  Found existing model: {EFFICIENTNET_MODEL_PATH}")
        response = input("Continue training will overwrite this model. Proceed? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
   
    # Load data
    train_loader, test_loader = get_dataloaders()
    
    # Build model
    print(f"\n🏗️  Building EfficientNet-B0...")
    model = build_model()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LR, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
        verbose=True
    )

    # Mixed Precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=USE_MIXED_PRECISION)

    # Training state
    best_f1 = 0.0
    patience_counter = 0
    training_history = defaultdict(list)
   
    # Print configuration
    print(f"\n{'='*70}")
    print(f"🚀 Training Configuration:")
    print(f"   Device: {DEVICE}")
    print(f"   Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"   Batch Size: {BATCH_SIZE} (train) / {VAL_BATCH_SIZE} (val)")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Learning Rate: {LR}")
    print(f"   Weight Decay: {WEIGHT_DECAY}")
    print(f"   Mixed Precision: {USE_MIXED_PRECISION}")
    print(f"   Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS} steps")
    print(f"   Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"   Workers: {NUM_WORKERS}")
    print(f"{'='*70}\n")

    # Training loop
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        
        epoch_start = time.time()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # Forward pass với mixed precision
            with torch.amp.autocast('cuda', enabled=USE_MIXED_PRECISION):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
           
            # Backward pass
            scaler.scale(loss).backward()
            
            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Metrics
            running_loss += loss.item() * images.size(0) * GRADIENT_ACCUMULATION_STEPS
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
           
            # Update progress bar
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item()*GRADIENT_ACCUMULATION_STEPS:.4f}',
                    'acc': f'{100.*correct/total:.1f}%'
                })
            
            # Clear cache periodically
            if batch_idx % CLEAR_CACHE_EVERY_N_BATCHES == 0:
                torch.cuda.empty_cache()

        train_loss = running_loss / total
        train_acc = correct / total
        epoch_time = time.time() - epoch_start

        # Evaluate on validation set
        val_loss, metrics = evaluate(model, test_loader, criterion)

        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(metrics['accuracy'])
        training_history['val_f1'].append(metrics['f1_score'])

        # Print epoch results
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{NUM_EPOCHS} - {epoch_time:.1f}s")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f}, acc={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1_score']:.4f}, AUC={metrics['auc_roc']:.4f}")
       
        # Learning rate scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(metrics['f1_score'])
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"  📉 LR reduced: {old_lr:.6f} -> {new_lr:.6f}")
        else:
            print(f"  LR: {new_lr:.6f}")

        # Save best model
        if metrics["f1_score"] > best_f1 + MIN_DELTA:
            improvement = metrics["f1_score"] - best_f1
            best_f1 = metrics["f1_score"]
            patience_counter = 0
            torch.save(model.state_dict(), EFFICIENTNET_MODEL_PATH)
            print(f"  ✅ BEST MODEL SAVED! F1={best_f1:.4f} (+{improvement:.4f})")
        else:
            patience_counter += 1
            print(f"  ⚠️  No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
           
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n{'='*70}")
                print(f"🛑 Early Stopping triggered at epoch {epoch}")
                print(f"   Best F1-Score: {best_f1:.4f}")
                print(f"{'='*70}\n")
                break
        
        # Save checkpoint periodically
        if epoch % SAVE_CHECKPOINT_EVERY == 0:
            checkpoint_path = EFFICIENTNET_MODEL_PATH.parent / f"checkpoint_epoch_{epoch}.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, metrics, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path.name}")
       
        # Memory management
        if epoch % GC_COLLECT_EVERY_N_EPOCHS == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"  💾 GPU Memory: {mem_allocated:.2f} GB / {mem_total:.2f} GB")

    # Training summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ Training completed!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Best F1-Score: {best_f1:.4f}")
    print(f"   Model saved: {EFFICIENTNET_MODEL_PATH}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    train()