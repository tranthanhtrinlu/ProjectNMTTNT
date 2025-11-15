package spring.api.java.projectai.service; // Đảm bảo dòng package này đúng

import org.springframework.core.io.InputStreamResource;
import java.io.InputStream;

public class MultipartInputStreamFileResource extends InputStreamResource {

    private final String filename;

    public MultipartInputStreamFileResource(InputStream inputStream, String filename) {
        super(inputStream);
        this.filename = filename;
    }

    @Override
    public String getFilename() {
        return this.filename;
    }

    @Override
    public long contentLength() {
        return -1; // để Spring không cố tính size
    }
}