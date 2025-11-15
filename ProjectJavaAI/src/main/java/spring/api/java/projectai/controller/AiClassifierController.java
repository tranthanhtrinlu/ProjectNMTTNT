package spring.api.java.projectai.controller;

import spring.api.java.projectai.model.PredictionResponse;
import spring.api.java.projectai.service.AiClassifierService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * Controller xử lý:
 * - GET "/"  → hiển thị trang upload hình
 * - POST "/classify" → nhận hình, gọi AI service, trả kết quả về JSP
 */
@Controller
public class AiClassifierController {

    private final AiClassifierService aiClassifierService;

    public AiClassifierController(AiClassifierService aiClassifierService) {
        this.aiClassifierService = aiClassifierService;
    }

    @GetMapping("/")
    public String index() {
        // Trả về file /WEB-INF/views/index.jsp
        return "index";
    }

    @PostMapping("/classify")
    public String classify(@RequestParam("image") MultipartFile file, Model model) {
        if (file.isEmpty()) {
            model.addAttribute("error", "Vui lòng chọn một ảnh để upload.");
            return "index";
        }

        try {
            PredictionResponse result = aiClassifierService.classifyImage(file);
            model.addAttribute("result", result);
            model.addAttribute("fileName", file.getOriginalFilename());
        } catch (Exception e) {
            model.addAttribute("error", "Gọi AI service thất bại: " + e.getMessage());
        }

        // Hiển thị lại index.jsp với dữ liệu result/error
        return "index";
    }
}