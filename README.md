## 📋 Giới thiệu

Dự án trích xuất tự động các đặc trưng từ ảnh phấn hoa sử dụng kỹ thuật xử lý ảnh truyền thống (OpenCV).

### Các đặc trưng được trích xuất

| Đặc trưng | Mô tả | Các giá trị |
|-----------|-------|-------------|
| **Shape** | Hình dạng hạt phấn | spherical, ellipsoidal, triangular, rectangular, irregular |
| **Size** | Kích thước (μm) | very_small (<10), small (10-25), medium (25-50), large (50-100), very_large (>100) |
| **Surface** | Bề mặt | psilate, scabrate, verrucate, echinate, reticulate, striate |
| **Aperture** | Lỗ mở | inaperturate, monocolpate, tricolpate, tricolporate, triporate, pantoporate |
| **Exine** | Lớp vỏ ngoài | thin, medium, thick, stratified |
| **Section** | Góc nhìn | equatorial, polar, oblique |

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
│   ├── feature_extractor.py   # Trích xuất đặc trưng
│   ├── database_handler.py    # Xử lý SQLite
│   ├── trainer.py             # Training (tùy chọn)
│   └── utils.py               # Hàm tiện ích
│
└── output/                    # Kết quả
    ├── extraction_results.json
    └── extraction_results.xlsx
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
```

### 2. Xuất kết quả ra Excel

```bash
python json_to_excel.py
```

File Excel sẽ chứa 4 sheets:
- **Chi tiết**: Dữ liệu đầy đủ từng ảnh
- **Tổng hợp theo Folder**: Thống kê tóm tắt mỗi folder
- **Chi tiết theo Folder**: Đếm chi tiết từng loại
- **Tổng quan**: Thống kê tổng thể

### 3. Sử dụng trong code Python

```python
from pollen_features import PollenFeatureExtractor

extractor = PollenFeatureExtractor()
features = extractor.extract_all_features("image.jpg")

print(f"Shape: {features['shape']['shape_class']}")
print(f"Size: {features['size']['size_full']}")  # VD: "45.2μm-medium"
print(f"Surface: {features['surface']['surface_class']}")
print(f"Aperture: {features['aperture_type']['aperture_class']}")
print(f"Exine: {features['exine']['exine_class']}")
print(f"Section: {features['section']['section_class']}")
```

## 🎓 Training Model

### Chuẩn bị dữ liệu

Tổ chức dữ liệu theo cấu trúc:
```
## 📊 Kết quả mẫu

```
📈 THỐNG KÊ THEO FOLDER:
   Folder 0: 216 ảnh
   Folder 1: 223 ảnh
   Folder 2: 489 ảnh
   ...
   Folder 25: 221 ảnh
   Tổng: 6,159 ảnh / 26 folders
```

## 🔧 Cấu hình

Chỉnh sửa `config.py`:

```python
# Thư mục ảnh đầu vào
DATA_DIR = r"C:\path\to\your\images"

# Kích thước ảnh xử lý
IMAGE_CONFIG = {
    'target_size': (128, 128),  # Giảm để tiết kiệm RAM
    'color_mode': 'rgb'
}
```

## 📝 Mô tả đặc trưng

| Đặc trưng | Phương pháp |
|-----------|-------------|
| **Shape** | Phân tích contour, fit ellipse |
| **Size** | Đo đường kính (pixels → μm) |
| **Surface** | Phân tích texture, LBP |
| **Aperture** | Edge detection, đếm lỗ mở |
| **Exine** | Phân tích gradient biên |
| **Section** | Phân tích đối xứng |

## 📄 License

MIT License

