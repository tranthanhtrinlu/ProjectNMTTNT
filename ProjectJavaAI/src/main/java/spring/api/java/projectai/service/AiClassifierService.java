package spring.api.java.projectai.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import spring.api.java.projectai.model.PredictionResponse;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

@Service
public class AiClassifierService {

    @Value("${ai.service.url}")   // ví dụ: http://localhost:8000/predict
    private String aiServiceUrl;

    private final RestTemplate restTemplate;

    public AiClassifierService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public PredictionResponse classifyImage(MultipartFile file) throws IOException {

        // 1. Lưu file upload ra file tạm
        String originalName = file.getOriginalFilename();
        Path temp = Files.createTempFile("upload-", "-" + (originalName != null ? originalName : "image"));
        Files.write(temp, file.getBytes());

        FileSystemResource resource = new FileSystemResource(temp.toFile());

        // 2. Tạo body multipart với KEY = "file"
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", resource);   // <<=== TÊN FIELD "file" bắt buộc trùng với FastAPI

        // 3. Header cho request
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        HttpEntity<MultiValueMap<String, Object>> requestEntity =
                new HttpEntity<>(body, headers);

        // 4. Gửi sang FastAPI
        ResponseEntity<PredictionResponse> response = restTemplate.exchange(
                aiServiceUrl,
                HttpMethod.POST,
                requestEntity,
                PredictionResponse.class
        );

        // (tuỳ anh có muốn xoá file tạm sau khi dùng xong không)
        try {
            Files.deleteIfExists(temp);
        } catch (IOException ignored) {}

        if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
            return response.getBody();
        }

        throw new RuntimeException("Lỗi gọi AI service: " + response.getStatusCode());
    }
}
