<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<html>
<head>
    <meta charset="UTF-8">
    <title>AI Image Real/Fake Classifier</title>
    <style>
        /* Toàn bộ body căn giữa theo cả trục dọc và ngang */
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f9f3ff;
            color: #4b0082;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }

        /* Card chứa form và kết quả */
        .container {
            background-color: #fff;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(155, 89, 182, 0.3);
            width: 90%;
            max-width: 600px;
            text-align: center;
        }

        h1 {
            font-size: 2.5em;
            margin-bottom: 30px;
        }

        h2 {
            font-size: 2em;
            margin-top: 20px;
            margin-bottom: 15px;
        }

        h3 {
            font-size: 1.6em;
            margin-top: 15px;
            margin-bottom: 10px;
        }

        form {
            margin-bottom: 30px;
        }

        label {
            display: block;
            font-size: 1.3em;
            margin-bottom: 10px;
        }

        input[type="file"] {
            padding: 12px;
            margin: 10px 0 20px 0;
            border-radius: 5px;
            border: 1px solid #ccc;
            font-size: 1.1em;
        }

        button {
            background-color: #9b59b6;
            color: white;
            border: none;
            padding: 12px 25px;
            font-size: 1.2em;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        button:hover {
            background-color: #8e44ad;
        }

        hr {
            margin: 30px 0;
            border: 0;
            border-top: 2px solid #dcd0f0;
        }

        p, li {
            font-size: 1.2em;
            margin: 5px 0;
        }

        ul {
            list-style-type: none;
            padding: 0;
        }

        /* Box hiển thị metrics */
        .metrics {
            display: inline-block;
            text-align: left;
            margin-top: 10px;
            padding: 15px 25px;
            background-color: #e8d6f4;
            border-radius: 12px;
        }

        .metrics li {
            font-weight: bold;
        }

        /* Lỗi hiển thị màu đỏ */
        .error {
            color: #c0392b;
            font-weight: bold;
            font-size: 1.3em;
        }

        /* Ảnh hiện thị */
        img.uploaded {
            max-width: 100%;
            margin-top: 15px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(155, 89, 182, 0.5);
        }

        /* CSS cho kết quả */
    </style>
</head>

<body>
<div class="container">
    <h1>AI Image Real/Fake Classifier</h1>

    <form action="${pageContext.request.contextPath}/classify" method="post" enctype="multipart/form-data">
        <label>Chọn ảnh để phân loại:</label>
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Phân loại</button>
    </form>

    <hr/>

    <!-- CHỈ HIỂN THỊ ERROR KHI CÓ -->
    <c:if test="${not empty error}">
        <p style="color:red;">${error}</p>
    </c:if>

    <!-- CHỈ HIỂN THỊ KẾT QUẢ KHI CÓ -->
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
</div>
</body>

</html>
