# Document Intelligence & Spatial KIE Pipeline

## Day 3: Web App, API & Docker

### Chạy local

Cài dependencies bằng Python 3.10+:

```powershell
python -m pip install -r requirements.txt
```

API đồng thời phục vụ giao diện web tại `/`, tải model PaddleOCR khi khởi động lần đầu và warm-up bằng ảnh 64x64:

```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Mở [http://127.0.0.1:8000](http://127.0.0.1:8000). Giao diện trong `receipt-web/` được phục vụ cùng origin với API nên không cần cấu hình CORS.

### API

- `GET /`: giao diện web Receipt Intelligence.
- `GET /health`: kiểm tra API đang hoạt động.
- `POST /api/v1/receipts/process`: nhận multipart field `image` (JPEG/PNG), chạy pipeline, lưu kết quả vào SQLite và trả về ID bản ghi cùng OCR boxes, ảnh đã xử lý và latency từng pha.
- `GET /api/v1/receipts`: danh sách phân trang; hỗ trợ `date_from`, `date_to`, `merchant`, `status`, `page`, `page_size`.
- `GET /api/v1/receipts/{id}`: chi tiết hóa đơn và dữ liệu trích xuất.
- `PUT /api/v1/receipts/{id}`: sửa `merchant`, `date` (ISO `YYYY-MM-DD`), `total`, `address`, `status`; mỗi thay đổi được ghi vào audit log. Có thể gửi `X-Actor` để lưu danh tính người sửa.
- `POST /analyze`: endpoint tương thích, không lưu dữ liệu.

SQLite mặc định lưu tại `outputs/receipts.db`; có thể đổi bằng `DATABASE_PATH` hoặc cấu hình `DATABASE_URL`. SHA-256 bắt bản trùng byte-for-byte; pHash bắt ảnh gần giống với ngưỡng Hamming mặc định 4. Trùng lặp trả HTTP 409, mã `DUPLICATE_RECEIPT_DETECTED` và ID bản cũ. Tài liệu Swagger tại `/docs`.

Upload bị giới hạn ở 10 MB; ảnh có cạnh lớn hơn 4096 px hoặc quá 16 megapixel bị từ chối trước khi giải mã bằng OpenCV. MIME type phải khớp định dạng ảnh thực tế. Inference được tuần tự hóa để tránh gọi đồng thời trên cùng OCR engine.

### Docker

Build image (bước build tải trước model OCR vào cache của image):

```powershell
docker build -t document-intelligence-kie .
docker run --rm -p 8000:8000 -v receipt-data:/app/outputs document-intelligence-kie
```

Container phục vụ cả giao diện web và API tại `http://localhost:8000`.

## Bảng kết quả Benchmark & Ablation Study (Day 2)

### 1. Kết quả trên tập Holdout `test` (30 hóa đơn)
| Cấu hình Pipeline | Word Error Rate (WER) ↓ | Entity F1 - Date ↑ | Entity F1 - Total ↑ | Overall KIE F1 ↑ | Latency (ms/ảnh) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Raw Image + OCR + Regex thuần)** | 50.19% | 86.79% | 20.00% | 52.38% | 19775.5 ms |
| **Full Pipeline (OpenCV + OCR + Spatial KIE)** | **27.47%** | **88.46%** | **85.19%** | **78.90%** (+26.52%) | 19736.9 ms |

### 2. Kết quả trên tập `test_degraded` (Xoay nghiêng ngẫu nhiên ±15° + Phủ bóng tối điện thoại)
| Cấu hình Pipeline | Word Error Rate (WER) ↓ | Entity F1 - Date ↑ | Entity F1 - Total ↑ | Overall KIE F1 ↑ | Latency (ms/ảnh) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Raw Image + OCR + Regex thuần)** | 53.07% | 86.27% | 6.90% | 50.91% | 20607.7 ms |
| **Ablation (Raw Image + OCR + Spatial KIE)** | 40.32% | 86.27% | 33.33% | 60.19% | 20633.1 ms |
| **Full Pipeline (OpenCV + OCR + Spatial KIE)** | **30.27%** | **86.27%** | **81.48%** | **74.65%** (+23.74%) | 21929.6 ms |
