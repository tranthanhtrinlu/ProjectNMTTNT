<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%@ taglib prefix="c" uri="jakarta.tags.core" %>
<%@ taglib prefix="fmt" uri="jakarta.tags.fmt" %>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Detector | Christmas Edition 🎄</title>
    <link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@500;700&display=swap" rel="stylesheet">
    <style>
        /* --- 1. Cấu trúc & Nền Noel --- */
        body {
            font-family: 'Quicksand', sans-serif;
            background: linear-gradient(135deg, #165b33 0%, #ca2323 100%); /* Gradient Xanh - Đỏ */
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
            overflow-x: hidden;
            position: relative;
        }

        /* Hiệu ứng tuyết rơi (CSS thuần) */
        .snowflake {
            color: #fff;
            font-size: 1em;
            font-family: Arial, sans-serif;
            text-shadow: 0 0 5px #000;
            position: fixed;
            top: -10%;
            z-index: 0;
            user-select: none;
            cursor: default;
            animation-name: snowflakes-fall, snowflakes-shake;
            animation-duration: 10s, 3s;
            animation-timing-function: linear, ease-in-out;
            animation-iteration-count: infinite, infinite;
            animation-play-state: running, running;
        }
        @keyframes snowflakes-fall { 0% { top: -10%; } 100% { top: 100%; } }
        @keyframes snowflakes-shake { 0%, 100% { transform: translateX(0); } 50% { transform: translateX(80px); } }

        .snowflake:nth-of-type(1) { left: 1%; animation-delay: 0s, 0s; }
        .snowflake:nth-of-type(2) { left: 10%; animation-delay: 1s, 1s; }
        .snowflake:nth-of-type(3) { left: 20%; animation-delay: 6s, .5s; }
        .snowflake:nth-of-type(4) { left: 30%; animation-delay: 4s, 2s; }
        .snowflake:nth-of-type(5) { left: 40%; animation-delay: 2s, 2s; }
        .snowflake:nth-of-type(6) { left: 50%; animation-delay: 8s, 3s; }
        .snowflake:nth-of-type(7) { left: 60%; animation-delay: 6s, 2s; }
        .snowflake:nth-of-type(8) { left: 70%; animation-delay: 2.5s, 1s; }
        .snowflake:nth-of-type(9) { left: 80%; animation-delay: 1s, 0s; }
        .snowflake:nth-of-type(10) { left: 90%; animation-delay: 3s, 1.5s; }

        /* --- 2. Container chính (Thẻ quà tặng) --- */
        .container {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 2.5rem;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            width: 100%;
            max-width: 550px;
            text-align: center;
            position: relative;
            z-index: 1; /* Nổi lên trên tuyết */
            border: 4px solid #f1c40f; /* Viền vàng sang trọng */
        }

        /* Trang trí góc container */
        .decoration {
            position: absolute;
            font-size: 2.5rem;
            top: -25px;
        }
        .dec-left { left: -15px; transform: rotate(-20deg); }
        .dec-right { right: -15px; transform: rotate(20deg); }

        h1 {
            color: #c0392b; /* Đỏ Noel */
            margin-bottom: 1.5rem;
            font-size: 2rem;
            font-weight: 700;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }

        /* --- 3. Upload Area --- */
        .upload-wrapper {
            position: relative;
            margin-bottom: 20px;
        }

        .file-input {
            position: absolute;
            width: 0.1px;
            height: 0.1px;
            opacity: 0;
            overflow: hidden;
            z-index: -1;
        }

        .file-label {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 30px;
            border: 3px dashed #27ae60; /* Viền xanh lá */
            border-radius: 15px;
            background-color: #f0fff4;
            cursor: pointer;
            transition: all 0.3s;
            color: #2c3e50;
        }

        .file-label:hover {
            border-color: #c0392b; /* Hover đổi sang đỏ */
            background-color: #fff5f5;
            transform: scale(1.02);
        }

        .icon-upload {
            font-size: 3rem;
            margin-bottom: 10px;
        }

        /* --- 4. Nút bấm --- */
        .btn-submit {
            background: linear-gradient(to right, #e74c3c, #c0392b);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            font-size: 1.2rem;
            font-weight: bold;
            width: 100%;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.4);
            margin-top: 10px;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(231, 76, 60, 0.6);
            background: linear-gradient(to right, #c0392b, #a93226);
        }

        /* --- 5. Preview Ảnh --- */
        #image-preview-container {
            margin-top: 20px;
            /* Logic: Nếu có ảnh base64 từ server hoặc JS preview thì hiện */
            display: ${not empty base64Image ? 'block' : 'none'};
            text-align: center;
            position: relative;
        }

        #image-preview {
            max-width: 100%;
            max-height: 300px;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            border: 3px solid white;
        }

        /* --- 6. Kết quả --- */
        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            background: #fff;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            border: 2px solid #ecf0f1;
        }

        @keyframes popIn {
            from { opacity: 0; transform: scale(0.8); }
            to { opacity: 1; transform: scale(1); }
        }

        .result-item {
            margin-bottom: 12px;
            font-size: 1.1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px dashed #eee;
            padding-bottom: 8px;
        }

        .label-fake {
            color: #e74c3c;
            font-weight: 800;
            font-size: 1.4rem;
            background: #ffebeb;
            padding: 5px 15px;
            border-radius: 20px;
        }
        .label-real {
            color: #27ae60;
            font-weight: 800;
            font-size: 1.4rem;
            background: #e8f8f5;
            padding: 5px 15px;
            border-radius: 20px;
        }

        /* --- Loading Overlay --- */
        #loading-overlay {
            display: none;
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 20px;
            z-index: 20;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }
        .spinner {
            border: 6px solid #f3f3f3;
            border-top: 6px solid #c0392b; /* Đỏ */
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin-bottom: 15px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

    </style>
</head>
<body>

<div class="snowflake">❄</div><div class="snowflake">❅</div><div class="snowflake">❆</div>
<div class="snowflake">❄</div><div class="snowflake">❅</div><div class="snowflake">❆</div>
<div class="snowflake">❄</div><div class="snowflake">❅</div><div class="snowflake">❆</div>
<div class="snowflake">❄</div>

<div class="container">
    <div class="decoration dec-left">🎅</div>
    <div class="decoration dec-right">🎄</div>

    <div id="loading-overlay">
        <div class="spinner"></div>
        <p style="color: #c0392b; font-weight: bold; font-size: 1.2rem;">Ông già Noel đang soi ảnh... 🎅</p>
    </div>

    <h1>AI Image Detector</h1>

    <c:if test="${not empty error}">
        <div style="background:#fce4e4; color:#c0392b; padding:10px; border-radius:8px; margin-bottom:15px;">
            ⚠️ ${error}
        </div>
    </c:if>

    <form id="uploadForm" action="${pageContext.request.contextPath}/classify" method="post" enctype="multipart/form-data" onsubmit="showLoading()">

        <div class="upload-wrapper">
            <input type="file" id="fileInput" name="image" class="file-input" accept="image/*" required onchange="previewImage(event)">

            <label for="fileInput" class="file-label" id="drop-area">
                <span class="icon-upload">🎁</span>
                <span id="file-label-text">Chọn ảnh hoặc thả vào đây</span>
            </label>

            <div id="image-preview-container">
                <img id="image-preview"
                     src="${not empty base64Image ? 'data:image/jpeg;base64,'.concat(base64Image) : '#'}"
                     alt="Preview Image" />
            </div>
        </div>

        <button type="submit" class="btn-submit">🔍 Phân loại ngay</button>
    </form>

    <c:if test="${not empty result}">
        <div class="result-box">
            <h2 style="text-align: center; color: #2c3e50; font-size: 1.4rem; margin-top: 0;">📜 Kết quả soi chiếu</h2>

            <div class="result-item">
                <span>📁 Tên file</span>
                <span style="font-weight: bold; color: #555;">${fileName}</span>
            </div>

            <div class="result-item">
                <span>🤖 Model</span>
                <span style="background: #eee; padding: 2px 8px; border-radius: 4px; font-size: 0.9rem;">${result.model}</span>
            </div>

            <div class="result-item" style="border-bottom: none; margin-top: 15px; justify-content: center; flex-direction: column;">
                <div style="margin-bottom: 5px;">Dự đoán:</div>
                <span class="${result.label == 'Fake' ? 'label-fake' : 'label-real'}">
                        ${result.label}
                </span>
            </div>

            <div style="margin-top: 15px; background: #f9f9f9; padding: 10px; border-radius: 8px;">
                <div style="font-weight: bold; margin-bottom: 8px; color: #7f8c8d;">Chi tiết xác suất:</div>
                <c:forEach var="entry" items="${result.probabilities}">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>${entry.key}</span>
                        <div style="flex-grow: 1; margin: 0 10px; background: #ddd; height: 20px; border-radius: 10px; overflow: hidden; position: relative; top: 2px;">
                            <div style="background: ${entry.key == 'REAL' ? '#27ae60' : '#e74c3c'}; width: <fmt:formatNumber value="${entry.value}" type="percent"/>; height: 100%;"></div>
                        </div>
                        <strong style="color: #333;"><fmt:formatNumber value="${entry.value}" type="percent" maxFractionDigits="1" /></strong>
                    </div>
                </c:forEach>
            </div>
        </div>
    </c:if>
</div>

<script>
    function previewImage(event) {
        var input = event.target;
        var preview = document.getElementById('image-preview');
        var container = document.getElementById('image-preview-container');
        var labelText = document.getElementById('file-label-text');

        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                container.style.display = 'block';
                labelText.textContent = "🎁 Đã gói: " + input.files[0].name;
            }
            reader.readAsDataURL(input.files[0]);

            // Ẩn kết quả cũ khi chọn ảnh mới để đỡ rối
            var oldResult = document.querySelector('.result-box');
            if(oldResult) oldResult.style.display = 'none';
        }
    }

    function showLoading() {
        var input = document.getElementById('fileInput');
        // Nếu đã có file chọn mới hoặc đã có ảnh cũ (trường hợp user submit lại mà không chọn file mới - tuy nhiên input file thường reset)
        // Đơn giản là check input
        if (input.files.length > 0) {
            document.getElementById('loading-overlay').style.display = 'flex';
        }
    }
</script>

</body>
</html>