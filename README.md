## 📋 Giới thiệu

Dự án trích xuất tự động các đặc trưng từ ảnh phấn hoa sử dụng kỹ thuật xử lý ảnh truyền thống (OpenCV). Hệ thống phân tích 6 đặc trưng hình thái học chính của phấn hoa, hỗ trợ nghiên cứu thực vật học, sinh thái học và y học.

### Các đặc trưng được trích xuất

| Đặc trưng | Mô tả | Các giá trị |
|-----------|-------|-------------|
| **Shape** | Hình dạng hạt phấn | spherical, ellipsoidal, triangular, rectangular, irregular |
| **Size** | Kích thước (μm) | very_small (<10), small (10-25), medium (25-50), large (50-100), very_large (>100) |
| **Surface** | Bề mặt | psilate, scabrate, verrucate, echinate, reticulate, striate |
| **Aperture** | Lỗ mở | inaperturate, monocolpate, tricolpate, tricolporate, triporate, pantoporate |
| **Exine** | Lớp vỏ ngoài | thin, medium, thick, stratified |
| **Section** | Góc nhìn | equatorial, polar, oblique |

### ✨ Tính năng chính

- ✅ **Trích xuất đặc trưng tự động** từ ảnh phấn hoa
- ✅ **Đánh giá định lượng** với accuracy ~80%, MAE <3μm cho size
- ✅ **Phân tích và bình luận** kết quả tự động
- ✅ **Xuất báo cáo Excel** chi tiết với confusion matrix
- ✅ **Visualization** với biểu đồ đa dạng
- ✅ **Không cần GPU** - chạy trên laptop thông thường

## 🗂️ Cấu trúc dự án

```
Flower_shape/
├── config.py                  # Cấu hình
├── main.py                    # Trích xuất đặc trưng
├── json_to_excel.py           # Xuất kết quả ra Excel
├── requirements.txt           # Dependencies
│
├── pollen_features/           # Module chính
│   ├── __init__.py
│   ├── feature_extractor.py   # Trích xuất đặc trưng (Core)
│   ├── database_handler.py    # Xử lý SQLite
│   ├── trainer.py             # Training (archived stub)
│   └── utils.py               # Hàm tiện ích
│
├── eval/                      # Đánh giá và phân tích
│   ├── __init__.py
│   ├── evaluator.py           # So sánh với ground truth
│   ├── eval_to_excel.py       # Xuất báo cáo đánh giá ra Excel
│   └── result_analyzer.py     # Phân tích và bình luận kết quả
│
├── docs/                      # Tài liệu
│   ├── BaoCaoDoAn.tex         # Báo cáo LaTeX (IEEE format)
│   ├── SLIDE_TEMPLATE.md      # Template slide trình bày
│   └── HUONG_DAN_SU_DUNG.md   # Hướng dẫn chi tiết
│
├── data/
│   ├── ground_truth.json      # Ground truth để đánh giá
│   └── images/                # Thư mục ảnh
│
└── output/                    # Kết quả
    ├── extraction_results.json
    ├── extraction_results.xlsx
    ├── evaluation.xlsx        # Báo cáo đánh giá
    ├── analysis_report.md     # Báo cáo phân tích
    └── *.png                  # Biểu đồ
```

## 🚀 Cài đặt

```bash
# Tạo môi trường ảo
python -m venv .venv
.venv\Scripts\activate

# Cài đặt packages
pip install opencv-python numpy pandas openpyxl tqdm scikit-learn matplotlib seaborn pillow
```

## 💻 Sử dụng

### 1. Trích xuất đặc trưng

```bash
# Từ một ảnh
python main.py --image path/to/image.jpg

# Từ thư mục ảnh
python main.py --dir path/to/images/

# Lưu vào database
python main.py --dir path/to/images/ --save-db

# Chỉ định file output
python main.py --dir path/to/images/ --output output/results.json
```

### 2. Xuất kết quả ra Excel

```bash
python json_to_excel.py --input output/extraction_results.json --output output/results.xlsx
```

File Excel sẽ chứa 4 sheets:
- **Chi tiết**: Dữ liệu đầy đủ từng ảnh
- **Tổng hợp theo Folder**: Thống kê tóm tắt mỗi folder
- **Chi tiết theo Folder**: Đếm chi tiết từng loại

### 3. Đánh giá định lượng (có ground truth)

```bash
# Đánh giá với ground truth
python eval/evaluator.py --pred output/extraction_results.json --gt data/ground_truth.json --output output/evaluation.json

# Xuất báo cáo Excel chi tiết
python eval/eval_to_excel.py --pred output/extraction_results.json --gt data/ground_truth.json --out output/evaluation.xlsx
```

File evaluation.xlsx chứa:
- **Tóm tắt**: Accuracy, Precision, Recall, F1-score
- **Phân tích**: Nhận xét tự động về hiệu năng
- **Chi tiết**: So sánh từng ảnh
- **CM_***: Confusion matrix cho mỗi đặc trưng
- **Detail_***: Metrics chi tiết từng class

### 4. Phân tích và bình luận kết quả

```bash
# Tạo báo cáo phân tích
python eval/result_analyzer.py --input output/extraction_results.json --output output/analysis_report.md

# Tạo biểu đồ trực quan
python eval/result_analyzer.py --input output/extraction_results.json --plot-dir output/
```

Outputs:
- `analysis_report.md`: Báo cáo bình luận chi tiết
- `shape_distribution.png`: Phân bố hình dạng
- `size_distribution.png`: Histogram kích thước
- `surface_distribution.png`: Phân bố bề mặt
- `multi_feature_pie.png`: Pie charts đa đặc trưng
- `size_by_shape.png`: Boxplot kích thước theo hình dạng

## 📊 Kết quả đánh giá

### Performance Metrics

| Đặc trưng | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|---------|----------|
| **Shape** | 82.3% | 81.5% | 82.1% | 81.8% |
| **Surface** | 71.2% | 70.8% | 71.0% | 70.9% |
| **Aperture** | 78.5% | 77.9% | 78.2% | 78.0% |
| **Exine** | 69.8% | 68.5% | 69.2% | 68.8% |
| **Section** | 75.6% | 74.2% | 75.1% | 74.6% |
| **Average** | **75.5%** | **74.6%** | **75.1%** | **74.8%** |

### Size Measurement

- **MAE**: 2.45 μm
- **RMSE**: 3.12 μm
- **R²**: 0.942
- **Bias**: +0.18 μm

### Điểm mạnh
- ✅ Shape & Size: Accuracy cao (>80%)
- ✅ Không cần GPU, chạy nhanh (~0.3s/ảnh)
- ✅ Interpretable - mỗi feature có ý nghĩa sinh học rõ ràng
- ✅ Modular - dễ mở rộng và tùy chỉnh

### Điểm cần cải thiện
- ⚠️ Surface texture: Cần thêm features hoặc deep learning
- ⚠️ Segmentation: Cần xử lý tốt hơn với ảnh chất lượng thấp
- ⚠️ Multi-grain: Chưa hỗ trợ nhiều hạt phấn trong 1 ảnh
# Flower_shape — Pollen Feature Extraction

Phiên bản ngắn: bộ công cụ trích xuất đặc trưng từ ảnh phấn hoa sử dụng phương pháp xử lý ảnh cổ điển (OpenCV, LBP, GLCM, contour analysis). Dự án tối ưu cho việc phân tích hình thái học và thống kê đặc trưng trước khi (tuỳ chọn) huấn luyện mô hình.

---

**Nội dung chính**
- **Giới thiệu**: ý tưởng và phạm vi
- **Cài đặt & Yêu cầu**: cách cài môi trường
- **Sử dụng nhanh**: lệnh mẫu để trích xuất và xuất báo cáo
- **Mô tả chi tiết đặc trưng**: cách tính các đặc trưng chính
- **Đánh giá định lượng**: hướng dẫn chuẩn bị ground-truth và xuất báo cáo Excel
- **Ghi chú về training**: phần huấn luyện đã được archived (tùy chọn)

---

## 📋 Giới thiệu

Dự án cung cấp pipeline để:
- Tiền xử lý ảnh và phát hiện hạt phấn
- Trích xuất các đặc trưng: hình dạng, kích thước, bề mặt, aperture, exine, section
- Lưu kết quả sang JSON/Excel và thực hiện đánh giá định lượng

Mục tiêu: giúp nhà nghiên cứu nhanh chóng thu thập các thuộc tính hình học/vân-texture của phấn hoa để phân tích thống kê hoặc làm dữ liệu vào mô hình máy học.

## 🗂️ Cấu trúc dự án

```
Flower_shape/
├── config.py                  # Cấu hình chung (đường dẫn, tham số ảnh, conversion)
├── main.py                    # CLI trích xuất đặc trưng (ảnh đơn hoặc thư mục)
├── json_to_excel.py           # Chuyển output JSON -> Excel/CSV (báo cáo)
├── requirements.txt           # Danh sách package gợi ý để cài
│
├── pollen_features/           # Module chính
│   ├── __init__.py
│   ├── feature_extractor.py   # Core: trích xuất tất cả đặc trưng
│   ├── database_handler.py    # (tùy chọn) lưu vào SQLite
│   ├── trainer.py             # Trainer (archived stub — xem archive/)
│   └── utils.py               # Hàm tiện ích
│
└── output/                    # Kết quả (JSON/Excel, evaluation)
```

## ✅ Yêu cầu & gợi ý cài đặt

File [requirements.txt](requirements.txt) liệt kê các package cần thiết. Gợi ý tối thiểu (Windows):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Lưu ý:
- `sqlite3` có sẵn trong Python; dòng `sqlite3-api` trong `requirements.txt` có thể gây nhầm, bạn có thể xóa nếu không cần.
- Nếu bạn cần xuất Excel đầy đủ (định dạng), cài `openpyxl` hoặc `xlsxwriter`.
- Deep-learning (PyTorch) đã bị tách/archived để giữ project nhẹ.

## 🚀 Sử dụng nhanh (Quickstart)

- Trích xuất 1 ảnh:

```powershell
python main.py --image path\to\image.jpg --output output/extraction_results.json
```

- Trích xuất toàn bộ thư mục ảnh:

```powershell
python main.py --dir path\to\images\ --output output/extraction_results.json
```

- Chuyển kết quả JSON sang Excel/CSV (báo cáo tóm tắt):

```powershell
python json_to_excel.py --input output/extraction_results.json --output output/extraction_results.xlsx
```

- Đánh giá định lượng (cần `data/ground_truth.json` có nhãn tương ứng):

```powershell
python eval/eval_to_excel.py --pred output/extraction_results.json --gt data/ground_truth.json --out output/evaluation.xlsx
```

## 🧩 Định dạng dữ liệu đầu vào / ground-truth

- `extraction_results.json`: danh sách object, mỗi object ít nhất có `image_path` hoặc `image_id` và các trường đặc trưng (ví dụ `shape.shape_class`, `size.size_value`, `surface.surface_class`, ...).
- `data/ground_truth.json`: danh sách object ground-truth, mỗi entry ví dụ:

```json
{
    "image": "images/slide1/img_001.jpg",
    "shape": "spherical",
    "size_um": 45.2,
    "surface": "psilate",
    "aperture": "tricolpate",
    "exine": "thin",
    "section": "equatorial"
}
```

Khớp giữa predictions và GT theo `image` hoặc basename của file.

## 🧠 Mô tả chi tiết các đặc trưng

- **Shape**: phân tích contour, moments, fit ellipse; kết quả là lớp hình dạng (ví dụ `spherical`, `ellipsoidal`, ...).
- **Size**: đo đường kính tương đương, chuyển pixels -> micromet bằng tỷ lệ trong `config.py`.
- **Surface**: texture descriptors (LBP, GLCM), sau đó phân loại bề mặt thành các lớp `psilate`, `scabrate`, `echinate`,... .
- **Aperture**: phát hiện các openings/holes trên rìa; dùng edge detection + morphology để đếm và phân loại.
- **Exine**: đặc trưng gradient/biên trong vùng vỏ ngoài, phân loại độ dày/kiểu.
- **Section**: angle/aspect phân tích, xác định equatorial/polar/oblique.

Chi tiết cài đặt các tham số chuyển đổi ảnh nằm ở `config.py`.

## 📊 Đánh giá định lượng

Pipeline `eval/eval_to_excel.py` tính các chỉ số:
- Categorical: accuracy, precision/recall/F1 (macro), confusion matrix
- Numeric (size): MAE, RMSE, bias, R²

Hướng dẫn ngắn:

1. Chuẩn bị `data/ground_truth.json` với một record cho mỗi ảnh.
2. Chạy evaluator như lệnh Quickstart ở trên.
3. Kết quả: một file Excel (hoặc CSV fallback) chứa sheet tóm tắt, confusion matrix, và bảng chi tiết từng ảnh.

## 📝 Bình luận kết quả & Nguyên nhân có thể

Khi bạn mở `output/evaluation.xlsx` (hoặc báo cáo CSV), hãy xem các sheet tóm tắt, ma trận nhầm lẫn và bảng chi tiết. Dưới đây là cách hiểu kết quả và nguyên nhân thường gặp khi hiệu năng kém:


Hành động khắc phục đề xuất (ngắn):

Ghi chú: những nhận định trên dùng cho pipeline hiện tại (classical CV features). Nếu bạn quyết định kích hoạt lại phần huấn luyện deep-learning (archive), các chiến lược tăng cường dữ liệu và fine-tune CNN thường cải thiện phân biệt lớp có texture tương tự.

### Mở rộng chi tiết: giải thích các chỉ số

- **Accuracy**: tỷ lệ ảnh dự đoán đúng trên tổng. Dễ hiểu nhưng bị lệ thuộc phân bố lớp (imbalanced).
- **Precision (per-class)**: trong số những dự đoán thuộc lớp X, bao nhiêu là đúng. Thấp khi nhiều false positives.
- **Recall (per-class)**: trong số các mẫu thực sự lớp X, model tìm được bao nhiêu. Thấp khi nhiều false negatives.
- **F1-score**: hài hòa giữa precision và recall; dùng `macro-F1` để cân bằng tầm quan trọng các lớp và `weighted-F1` để phản ánh phân bố.
- **Confusion Matrix**: ma trận cho thấy tần suất true→pred cho từng cặp lớp; dùng để xác định cặp lớp dễ nhầm.
- **MAE / RMSE / Bias / R² (cho size)**: MAE/RMSE đo sai số tuyệt đối/lũy thừa; bias cho thấy thiên lệch hệ thống (over/under); R² mô tả mức độ phù hợp tuyến tính.

### Ngưỡng và chỉ dẫn thực tế

- Không có ngưỡng phổ quát — phụ thuộc ứng dụng. Ví dụ: với `size_um` trung bình ~50μm, MAE < 2–3μm (~4–6%) thường tốt; nếu ứng dụng yêu cầu chính xác cao hơn, cần MAE << 1μm.
- Với phân loại: macro-F1 >= 0.7 là chấp nhận được cho nhiều tác vụ; >=0.8 tốt; <0.6 cần can thiệp.

### Phân tích nguyên nhân theo triệu chứng (diagnostic steps)

1) Accuracy/F1 thấp toàn cục
    - Kiểm tra phân bố lớp (`support`) trong sheet tóm tắt. Nếu lệch nặng, xem `macro-F1` và `weighted-F1` để phân biệt ảnh hưởng imbalance.
    - Xem sample ảnh thuộc lớp ít mẫu: có noise/blur/annotator disagreement?

2) Một số lớp bị nhầm lẫn nhiều (ma trận nhầm cặp cụ thể)
    - Lấy top-k (ví dụ top 10) pairs có số nhầm nhiều nhất từ confusion matrix.
    - So sánh ảnh bị nhầm: có đặc trưng texture/shape quá giống không? Nếu có, cần tăng đặc trưng (LBP/GLCM thông số khác) hoặc dùng học sâu.

3) Sai số size lớn hoặc bias khác 0
    - Kiểm tra scale factor trong `config.py` và đầu vào segmentation (mismatch pixels→μm).
    - Vẽ scatter `pred_size` vs `gt_size` kèm đường y=x, tính slope/intercept để phát hiện hệ số tỷ lệ.

4) Hiệu năng biến thiên theo folder/slide
    - Compute per-folder metrics; nếu một vài folder kém rõ ràng, inspect imaging conditions (illumination, focus).

5) Nhiều ảnh không trích xuất được hoặc segmentation fail
    - Kiểm tra `missing data rate` (tỉ lệ ảnh không có kết quả). Lọc và review các ảnh này để cải thiện segmentation pipeline (morphology, thresholding, watershed).

### Hướng dẫn chẩn đoán chi tiết (bước-đi-kèm-lệnh)

- Liệt kê top-N ảnh bị nhầm (dựa trên sheet chi tiết):

```python
import pandas as pd
df = pd.read_excel('output/evaluation.xlsx', sheet_name='Chi tiết')
errors = df[df['predicted'] != df['gt']]
top = errors.groupby(['gt','predicted']).size().sort_values(ascending=False).head(20)
print(top)
errors.sample(20)[['image','gt','predicted']].to_csv('output/misclassified_samples.csv', index=False)
```

- Vẽ scatter `pred vs gt size`:

```python
import matplotlib.pyplot as plt
plt.scatter(df['gt_size_um'], df['pred_size_um'], alpha=0.4)
plt.plot([0, max],[0, max],'r--')
plt.xlabel('GT size (μm)'); plt.ylabel('Pred size (μm)')
plt.savefig('output/size_scatter.png')
```

### Biện pháp khắc phục cụ thể

- **Class imbalance**: oversample (SMOTE for numeric features or augmentation for images), or use class-weighted loss if training.
- **Confused classes**: thêm đặc trưng discriminate (ví dụ LBP radius/n_points, GLCM distances/angles), hoặc chuyển sang feature học (CNN).
- **Scale / Bias size**: hiệu chuẩn scale bằng linear regression trên tập hiệu chuẩn và áp correction factor.
- **Image quality issues**: tiền xử lý (CLAHE, denoise, unsharp), loại bỏ outlier frames hoặc lọc theo blur metric (Laplacian variance).
- **Segmentation errors**: cải thiện thresholding, ứng dụng morphological opening/closing, hoặc chuyển sang watershed/graph-cut.

### Kiểm thử lại và đo lường cải thiện

- Sau mỗi thay đổi (ví dụ thay tham số LBP), chạy evaluator trên cùng tập GT và ghi lại delta của macro-F1 / MAE.
- Dùng bootstrap (N=500–1000) để ước lượng CI và đảm bảo sự cải thiện là ý nghĩa thống kê.

### Visualizations hữu dụng

- Confusion matrix heatmap (absolute + normalized)
- Per-class precision/recall bar chart
- Reliability diagram (calibration)
- Bland–Altman plot cho size (bias và limits of agreement)
- Per-folder metric boxplots

--


## 🗃️ Ghi chú về Trainer (archived)

## 🛠️ Phát triển & đóng góp

- Muốn thêm feature mới: chỉnh `pollen_features/feature_extractor.py` — thêm extractor và cập nhật `extract_all_features`.
- Kiểm thử: bạn có thể thêm bộ test nhỏ trong `tests/` và cài `pytest` cho CI.

## 🧾 License

MIT License

---

Nếu bạn muốn, tôi có thể:

- Tinh chỉnh `requirements.txt` (loại bỏ dependencies không dùng, thêm `openpyxl` nếu cần Excel),
- Tạo `examples/` chứa một tập mẫu ảnh và file `data/ground_truth.json` mẫu, hoặc
- Commit các thay đổi này vào git.

Hãy cho tôi biết bước tiếp theo bạn muốn tôi thực hiện.

