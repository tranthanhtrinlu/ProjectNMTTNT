package spring.api.java.projectai.model;

import lombok.Data;
import java.util.Map;

/**
 * Dùng để map JSON trả về từ Python FastAPI:
 * {
 *   "model": "efficientnet_b0",
 *   "label": "REAL",
 *   "probabilities": { "FAKE": 0.12, "REAL": 0.88 }
 * }
 */

@Data
public class PredictionResponse {
    private String model;
    private String label;
    private Map<String, Double> probabilities;
}