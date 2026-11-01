# Document Intelligence & Spatial KIE Pipeline

## Day 3: API, Demo UI & Docker

### Chạy local

Cài dependencies bằng Python 3.10+:

```powershell
python -m pip install -r requirements.txt
```

Mở hai terminal tại thư mục dự án. API tải model PaddleOCR khi khởi động lần đầu và warm-up bằng ảnh 64x64:

```powershell
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

```powershell
streamlit run app/streamlit_app.py
```

Mở URL Streamlit được in trong terminal (mặc định `http://localhost:8501`). Có thể đặt `KIE_API_URL` nếu API không chạy tại `http://127.0.0.1:8000`.

### API

- `GET /health`: kiểm tra API đang hoạt động.
- `POST /analyze`: nhận multipart field `image` (JPEG/PNG), trả về `fields`, OCR `boxes` với polygon/text/confidence, logical lines, ảnh đã xử lý và overlay dạng Base64 cùng latency từng pha.

Upload bị giới hạn ở 10 MB; ảnh có cạnh lớn hơn 4096 px hoặc quá 16 megapixel bị từ chối trước khi giải mã bằng OpenCV. MIME type phải khớp định dạng ảnh thực tế. Inference được tuần tự hóa để tránh gọi đồng thời trên cùng OCR engine.

### Docker

Build image (bước build tải trước model OCR vào cache của image):

```powershell
docker build -t document-intelligence-kie .
docker run --rm -p 8000:8000 document-intelligence-kie
```

Container phục vụ API tại `http://localhost:8000`; chạy Streamlit local theo hướng dẫn trên và đặt `KIE_API_URL=http://127.0.0.1:8000` để dùng demo.

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
