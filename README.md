# Document Intelligence & Spatial KIE Pipeline

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
