import os

# Đường dẫn thư mục
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r"C:\Users\Huy\Downloads\Ha_0-25\Ha_0-25"  # Thư mục chứa ảnh phấn hoa
IMAGES_DIR = DATA_DIR  # Ảnh nằm trực tiếp trong thư mục này
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Tạo thư mục nếu chưa tồn tại
for dir_path in [OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Cấu hình xử lý ảnh
IMAGE_CONFIG = {
    "target_size": (256, 256), 
    "color_mode": "rgb",
    "normalize": True,
}

# Các loại đặc trưng phấn hoa
FEATURE_TYPES = {
    "shape": [
        "spherical",      # Hình cầu
        "ellipsoidal",    # Hình elip
        "triangular",     # Hình tam giác
        "rectangular",    # Hình chữ nhật
        "irregular",      # Bất thường
    ],
    "size": [
        "very_small",     # < 10 μm
        "small",          # 10-25 μm
        "medium",         # 25-50 μm
        "large",          # 50-100 μm
        "very_large",     # > 100 μm
    ],
    "surface": [
        "psilate",        # Nhẵn
        "scabrate",       # Sần nhẹ
        "verrucate",      # Có mụn
        "echinate",       # Có gai
        "reticulate",     # Mạng lưới
        "striate",        # Có vân
    ],
    "aperture_type": [
        "inaperturate",   # Không có lỗ
        "monocolpate",    # 1 rãnh
        "tricolpate",     # 3 rãnh
        "tricolporate",   # 3 rãnh có lỗ
        "triporate",      # 3 lỗ
        "pantoporate",    # Nhiều lỗ
    ],
    "exine": [
        "thin",           # Mỏng
        "medium",         # Trung bình
        "thick",          # Dày
        "stratified",     # Phân tầng
    ],
    "section": [
        "equatorial",     # Mặt cắt xích đạo
        "polar",          # Mặt cắt cực
        "oblique",        # Mặt cắt xiên
    ],
}

# Cấu hình model CNN
MODEL_CONFIG = {
    "backbone": "resnet50",  # Có thể thay bằng: vgg16, efficientnet, etc.
    "pretrained": True,
    "num_epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "dropout_rate": 0.5,
}

# Cấu hình database
DATABASE_CONFIG = {
    "path": os.path.join(DATA_DIR, "pollen_database.db"),
    "csv_export": os.path.join(OUTPUT_DIR, "features_export.csv"),
}
