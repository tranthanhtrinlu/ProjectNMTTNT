<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%@ taglib prefix="c" uri="jakarta.tags.core" %>
<html>
<head>
    <meta charset="UTF-8">
    <title>AI Image Real/Fake Classifier</title>
</head>
<body>
<h1>AI Image Real/Fake Classifier</h1>

<form action="${pageContext.request.contextPath}/classify" method="post" enctype="multipart/form-data">
    <label>Chọn ảnh để phân loại:</label>
    <input type="file" name="image" accept="image/*" required>
    <button type="submit">Phân loại</button>
</form>

<hr/>

<c:if test="${not empty error}">
    <p style="color:red;">${error}</p>
</c:if>

<c:if test="${not empty result}">
    <h2>Kết quả cho file: ${fileName}</h2>
    <p><strong>Model:</strong> ${result.model}</p>
    <p><strong>Label dự đoán:</strong> ${result.label}</p>

    <h3>Xác suất:</h3>
    <ul>
        <c:forEach var="entry" items="${result.probabilities}">
            <li>${entry.key}: ${entry.value}</li>
        </c:forEach>
    </ul>
</c:if>

</body>
</html>
