# Receipt Intelligence — giao diện web

Giao diện HTML/CSS/JS thuần, được FastAPI phục vụ tại `/` cùng origin với API.

## Chạy nhanh
Chạy backend từ thư mục gốc dự án:

```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000`. Bấm **Thử với ảnh mẫu** để xem giao diện mà chưa cần gọi OCR.

## Kết nối API
Giao diện gọi `POST /analyze` (field `image`) và `GET /health` cùng origin. Phản hồi gồm:
`fields`, `boxes`, `full_text`, `timings_ms`, `images.preprocessed_jpeg_base64`, `images.annotated_jpeg_base64`
(đúng theo schema của API). `fields` nhận cả dạng `{"total": "160.500"}`
lẫn `{"total": {"value", "confidence", "bbox"}}`; mỗi box có `bbox`, `text`, `confidence`, `is_key_value`.

Đặt `/` làm địa chỉ API trong phần Cài đặt nếu cần; khi chạy theo hướng dẫn mặc định, không cần thay đổi cài đặt kết nối.
