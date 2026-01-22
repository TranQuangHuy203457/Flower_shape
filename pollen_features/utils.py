import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union
import json
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def load_image(path: str, target_size: Tuple[int, int] = None) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if target_size:
        img = cv2.resize(img, target_size)
    
    return img


def save_image(image: np.ndarray, path: str):
    """Lưu ảnh ra file"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def visualize_features(image: np.ndarray, features: Dict, save_path: str = None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Đặc trưng phấn hoa', fontsize=14, fontweight='bold')
    
    # Ảnh gốc
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Ảnh gốc')
    axes[0, 0].axis('off')
    
    # Shape info
    shape = features.get('shape', {})
    shape_text = f"Shape: {shape.get('shape_class', 'N/A')}\n"
    shape_text += f"Confidence: {shape.get('confidence', 0):.2%}\n"
    if 'metrics' in shape:
        shape_text += f"Circularity: {shape['metrics'].get('circularity', 0):.3f}\n"
        shape_text += f"Aspect Ratio: {shape['metrics'].get('aspect_ratio', 0):.3f}"
    axes[0, 1].text(0.1, 0.5, shape_text, fontsize=12, transform=axes[0, 1].transAxes,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[0, 1].set_title('📐 Hình dạng')
    axes[0, 1].axis('off')
    
    # Size info
    size = features.get('size', {})
    size_text = f"Size: {size.get('size_class', 'N/A')}\n"
    if 'metrics' in size:
        size_text += f"Diameter: {size['metrics'].get('diameter_micron', 0):.2f} μm\n"
        size_text += f"Width: {size['metrics'].get('width_micron', 0):.2f} μm\n"
        size_text += f"Height: {size['metrics'].get('height_micron', 0):.2f} μm"
    axes[0, 2].text(0.1, 0.5, size_text, fontsize=12, transform=axes[0, 2].transAxes,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    axes[0, 2].set_title('📏 Kích thước')
    axes[0, 2].axis('off')
    
    # Surface info
    surface = features.get('surface', {})
    surface_text = f"Surface: {surface.get('surface_class', 'N/A')}\n"
    surface_text += f"Confidence: {surface.get('confidence', 0):.2%}"
    if 'metrics' in surface:
        surface_text += f"\nRoughness: {surface['metrics'].get('roughness', 0):.2f}"
    axes[1, 0].text(0.1, 0.5, surface_text, fontsize=12, transform=axes[1, 0].transAxes,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    axes[1, 0].set_title('🔍 Bề mặt')
    axes[1, 0].axis('off')
    
    # Aperture info
    aperture = features.get('aperture_type', {})
    aperture_text = f"Aperture: {aperture.get('aperture_class', 'N/A')}"
    if 'metrics' in aperture:
        aperture_text += f"\nNum apertures: {aperture['metrics'].get('num_apertures', 0)}"
    axes[1, 1].text(0.1, 0.5, aperture_text, fontsize=12, transform=axes[1, 1].transAxes,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    axes[1, 1].set_title('🕳️ Lỗ mở')
    axes[1, 1].axis('off')
    
    # Exine & Section info
    exine = features.get('exine', {})
    section = features.get('section', {})
    other_text = f"Exine: {exine.get('exine_class', 'N/A')}\n"
    other_text += f"Section: {section.get('section_class', 'N/A')}"
    axes[1, 2].text(0.1, 0.5, other_text, fontsize=12, transform=axes[1, 2].transAxes,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))
    axes[1, 2].set_title('🧱 Exine & Section')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Đã lưu visualization tại: {save_path}")
    
    plt.show()


def visualize_segmentation(image: np.ndarray, save_path: str = None):
    """Visualize các bước phân vùng phấn hoa"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Ảnh gốc')
    axes[0, 0].axis('off')
    
    # Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    axes[0, 2].imshow(blurred, cmap='gray')
    axes[0, 2].set_title('Gaussian Blur')
    axes[0, 2].axis('off')
    
    # Otsu threshold
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    axes[1, 0].imshow(binary, cmap='gray')
    axes[1, 0].set_title('Otsu Threshold')
    axes[1, 0].axis('off')
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    axes[1, 1].imshow(cleaned, cmap='gray')
    axes[1, 1].set_title('Morphological Cleaning')
    axes[1, 1].axis('off')
    
    # Contours overlay
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = image.copy()
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Detected Contours')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_sample_dataset(output_dir: str, num_samples: int = 10):
    import random
    
    os.makedirs(output_dir, exist_ok=True)
    
    shapes = {
        'spherical': lambda img, cx, cy, r: cv2.circle(img, (cx, cy), r, (100, 150, 200), -1),
        'ellipsoidal': lambda img, cx, cy, r: cv2.ellipse(img, (cx, cy), (r, int(r*0.6)), 0, 0, 360, (150, 100, 200), -1),
        'triangular': lambda img, cx, cy, r: cv2.drawContours(img, [np.array([[cx, cy-r], [cx-r, cy+r], [cx+r, cy+r]])], -1, (200, 150, 100), -1),
    }
    
    for shape_name, draw_func in shapes.items():
        shape_dir = os.path.join(output_dir, 'shape', shape_name)
        os.makedirs(shape_dir, exist_ok=True)
        
        for i in range(num_samples):
            # Tạo ảnh ngẫu nhiên
            img = np.random.randint(200, 255, (224, 224, 3), dtype=np.uint8)
            
            # Vẽ hình
            cx, cy = 112 + random.randint(-20, 20), 112 + random.randint(-20, 20)
            r = 60 + random.randint(-10, 10)
            draw_func(img, cx, cy, r)
            
            # Thêm texture ngẫu nhiên
            noise = np.random.randn(224, 224, 3) * 10
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            
            # Lưu
            save_path = os.path.join(shape_dir, f"{shape_name}_{i+1:03d}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print(f"Đã tạo dataset mẫu tại: {output_dir}")


def print_feature_summary(features: Dict):
    """In tóm tắt đặc trưng đẹp"""
    print("\n" + "="*50)
    print("           TÓM TẮT ĐẶC TRƯNG PHẤN HOA")
    print("="*50)
    
    summaries = [
        ("📐 Hình dạng", features.get('shape', {}).get('shape_class', 'N/A')),
        ("📏 Kích thước", features.get('size', {}).get('size_class', 'N/A')),
        ("🔍 Bề mặt", features.get('surface', {}).get('surface_class', 'N/A')),
        ("🕳️ Lỗ mở", features.get('aperture_type', {}).get('aperture_class', 'N/A')),
        ("🧱 Lớp vỏ", features.get('exine', {}).get('exine_class', 'N/A')),
        ("📷 Mặt cắt", features.get('section', {}).get('section_class', 'N/A')),
    ]
    
    for name, value in summaries:
        print(f"  {name:15} : {value}")
    
    print("="*50 + "\n")


def evaluate_feature_embeddings(X: np.ndarray, y: List, cv: int = 5) -> Dict:
    """Đánh giá định lượng chất lượng embedding/feature.

    Tham số:
    - X: mảng (n_samples, n_features)
    - y: danh sách nhãn tương ứng
    - cv: số fold cho cross-validation đánh giá phân loại

    Trả về một dict với các chỉ số: intra_class_distance, inter_class_distance,
    silhouette_score, classification_accuracy (mean/std), f1_macro (mean/std).
    """
    X = np.asarray(X)
    y = np.asarray(y)

    results = {}

    if X.size == 0 or len(X.shape) != 2:
        return {"error": "Invalid feature matrix X"}

    # Khoảng cách cặp
    D = pairwise_distances(X, metric='euclidean')

    # Tính intra-class và inter-class mean distances
    classes = np.unique(y)
    intra_dists = []
    inter_dists = []

    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) > 1:
            # lấy nửa trên của ma trận để tránh double counting
            sub = D[np.ix_(idx, idx)]
            triu_idx = np.triu_indices_from(sub, k=1)
            vals = sub[triu_idx]
            if vals.size > 0:
                intra_dists.append(vals.mean())

    # inter-class
    for i, ca in enumerate(classes):
        for cb in classes[i+1:]:
            ia = np.where(y == ca)[0]
            ib = np.where(y == cb)[0]
            if ia.size > 0 and ib.size > 0:
                vals = D[np.ix_(ia, ib)].ravel()
                inter_dists.append(vals.mean())

    results['intra_class_distance_mean'] = float(np.mean(intra_dists)) if intra_dists else None
    results['inter_class_distance_mean'] = float(np.mean(inter_dists)) if inter_dists else None

    # Silhouette score (n_classes >= 2)
    try:
        if len(classes) > 1 and X.shape[0] > len(classes):
            results['silhouette_score'] = float(silhouette_score(X, y))
        else:
            results['silhouette_score'] = None
    except Exception:
        results['silhouette_score'] = None

    # Đánh giá phân loại đơn giản với LogisticRegression (cross-val)
    try:
        clf = LogisticRegression(max_iter=2000)
        skf = StratifiedKFold(n_splits=min(cv, max(2, len(classes))), shuffle=True, random_state=0)
        acc_scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
        f1_scores = cross_val_score(clf, X, y, cv=skf, scoring='f1_macro')

        results['classification_accuracy_mean'] = float(np.mean(acc_scores))
        results['classification_accuracy_std'] = float(np.std(acc_scores))
        results['f1_macro_mean'] = float(np.mean(f1_scores))
        results['f1_macro_std'] = float(np.std(f1_scores))
    except Exception as e:
        results['classification_error'] = str(e)

    return results


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Tạo ảnh test
    test_img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
    cv2.circle(test_img, (112, 112), 80, (50, 100, 150), -1)
    
    # Test segmentation visualization
    print("Visualizing segmentation steps...")
    visualize_segmentation(test_img)
