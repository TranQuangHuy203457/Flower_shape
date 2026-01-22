import cv2
import numpy as np
from PIL import Image
 
import os
import gc
from typing import Dict, List, Tuple, Optional, Union

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGE_CONFIG, FEATURE_TYPES

# Tắt deep learning để tiết kiệm RAM - chỉ dùng xử lý ảnh truyền thống
USE_DEEP_LEARNING = False


class PollenFeatureExtractor:
    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = False):
        self.target_size = IMAGE_CONFIG["target_size"]
        self.feature_types = FEATURE_TYPES
        
        # Khởi tạo các bộ trích xuất đặc trưng (nhẹ, không dùng deep learning)
        self.shape_extractor = ShapeExtractor()
        self.size_extractor = SizeExtractor()
        self.surface_extractor = SurfaceExtractor()
        
        self.aperture_extractor = ApertureExtractor()
        self.exine_extractor = ExineExtractor()
        self.section_extractor = SectionExtractor()
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        # Đọc ảnh
        if isinstance(image, str):
            img_array = cv2.imread(image)
            if img_array is None:
                raise ValueError(f"Không thể đọc ảnh: {image}")
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Resize nếu target_size được chỉ định, nếu không giữ nguyên kích thước gốc
        if self.target_size is not None:
            img_resized = cv2.resize(img_array, self.target_size, interpolation=cv2.INTER_AREA)
        else:
            img_resized = img_array
        
        return img_resized
    
    def extract_all_features(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
        # Tiền xử lý ảnh
        img_array = self.preprocess_image(image)
        
        # Trích xuất từng loại đặc trưng
        features = {
            "shape": self.shape_extractor.extract(img_array),
            "size": self.size_extractor.extract(img_array),
            "surface": self.surface_extractor.extract(img_array),
            "aperture_type": self.aperture_extractor.extract(img_array),
            "exine": self.exine_extractor.extract(img_array),
            "section": self.section_extractor.extract(img_array),
        }
        
        # Giải phóng bộ nhớ
        del img_array
        gc.collect()
        
        return features
    
    def extract_batch(self, image_paths: List[str], batch_size: int = 10) -> List[Dict]:
        results = []
        
        for i, path in enumerate(image_paths):
            try:
                features = self.extract_all_features(path)
                features["image_path"] = path
                features["status"] = "success"
                results.append(features)
            except Exception as e:
                results.append({
                    "image_path": path,
                    "status": "error",
                    "error_message": str(e)
                })
            
            # Giải phóng bộ nhớ sau mỗi batch
            if (i + 1) % batch_size == 0:
                gc.collect()
        
        return results


class BaseFeatureExtractor:
    """Lớp cơ sở cho các bộ trích xuất đặc trưng"""
    
    def __init__(self):
        self.feature_name = "base"
    
    def extract(self, image: np.ndarray) -> Dict:
        """Phương thức trích xuất đặc trưng - cần được override"""
        raise NotImplementedError
    
    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Chuyển ảnh sang grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    def _segment_pollen(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = self._to_grayscale(image)
        
        # Làm mờ để giảm nhiễu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Ngưỡng Otsu
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations để làm sạch
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Tìm contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return binary, contours


class ShapeExtractor(BaseFeatureExtractor):
    """Trích xuất đặc trưng hình dạng của phấn hoa"""
    
    def __init__(self):
        super().__init__()
        self.feature_name = "shape"
        self.shape_classes = FEATURE_TYPES["shape"]
    
    def extract(self, image: np.ndarray) -> Dict:
        binary, contours = self._segment_pollen(image)
        
        if not contours:
            return {"shape_class": "unknown", "confidence": 0.0, "metrics": {}}
        
        # Lấy contour lớn nhất (giả định là hạt phấn)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Tính các đặc trưng hình học
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        # Circularity (độ tròn)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Fit ellipse
        if len(main_contour) >= 5:
            ellipse = cv2.fitEllipse(main_contour)
            (cx, cy), (major_axis, minor_axis), angle = ellipse
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 1
        else:
            aspect_ratio = 1
            major_axis = minor_axis = 0
        
        # Convexity
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Phân loại hình dạng
        shape_class, confidence = self._classify_shape(circularity, aspect_ratio, solidity)
        
        return {
            "shape_class": shape_class,
            "confidence": confidence,
            "metrics": {
                "area": float(area),
                "perimeter": float(perimeter),
            }
        }
    
    def _classify_shape(self, circularity: float, aspect_ratio: float, solidity: float) -> Tuple[str, float]:
        """Phân loại hình dạng dựa trên các metrics"""
        if circularity > 0.85 and aspect_ratio < 1.2:
            return "spherical", circularity
        elif 1.2 <= aspect_ratio <= 2.0 and circularity > 0.7:
            return "ellipsoidal", (1 - abs(aspect_ratio - 1.5) / 0.5) * circularity
        elif solidity < 0.85 and 2.5 <= self._count_vertices(circularity) <= 4:
            return "triangular", solidity
        elif aspect_ratio > 2.0:
            return "rectangular", 1 - circularity
        else:
            return "irregular", 1 - solidity
    
    def _count_vertices(self, circularity: float) -> int:
        """Ước tính số đỉnh dựa trên circularity"""
        # Công thức gần đúng
        if circularity > 0.9:
            return 100  # Gần như tròn
        return int(np.pi / (1 - circularity + 0.01))


class SizeExtractor(BaseFeatureExtractor):
    """Trích xuất đặc trưng kích thước của phấn hoa"""
    
    def __init__(self, pixel_to_micron: float = 0.5):
        super().__init__()
        self.feature_name = "size"
        self.size_classes = FEATURE_TYPES["size"]
        self.pixel_to_micron = pixel_to_micron
    
    def extract(self, image: np.ndarray) -> Dict:
        """Trích xuất đặc trưng kích thước"""
        binary, contours = self._segment_pollen(image)
        
        if not contours:
            return {"size_class": "unknown", "metrics": {}}
        
        main_contour = max(contours, key=cv2.contourArea)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Min enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(main_contour)
        diameter_pixels = 2 * radius
        
        # Chuyển sang micromet
        diameter_micron = diameter_pixels * self.pixel_to_micron
        width_micron = w * self.pixel_to_micron
        height_micron = h * self.pixel_to_micron
        
        # Phân loại kích thước
        size_class = self._classify_size(diameter_micron)
        
        # Tạo kết quả kích thước cụ thể: VD "23μm-small"
        size_value = f"{diameter_micron:.1f}μm"
        size_full = f"{diameter_micron:.1f}μm-{size_class}"
        
        return {
            "size_class": size_class,
            "size_value": size_value,
            "size_full": size_full,
            "metrics": {
                "diameter_pixels": float(diameter_pixels),
                "diameter_micron": float(diameter_micron),
                "width_micron": float(width_micron),
                "height_micron": float(height_micron),
                "area_pixels": float(cv2.contourArea(main_contour)),
            }
        }
    
    def _classify_size(self, diameter: float) -> str:
        """Phân loại kích thước theo đường kính (μm)"""
        if diameter < 10:
            return "very_small"
        elif diameter < 25:
            return "small"
        elif diameter < 50:
            return "medium"
        elif diameter < 100:
            return "large"
        else:
            return "very_large"
    
    def get_size_range(self, size_class: str) -> str:
        """Lấy khoảng kích thước cho mỗi loại"""
        ranges = {
            "very_small": "<10μm",
            "small": "10-25μm",
            "medium": "25-50μm",
            "large": "50-100μm",
            "very_large": ">100μm"
        }
        return ranges.get(size_class, "unknown")


class SurfaceExtractor(BaseFeatureExtractor):
    """Trích xuất đặc trưng bề mặt (texture) của phấn hoa"""
    
    def __init__(self):
        super().__init__()
        self.feature_name = "surface"
        self.surface_classes = FEATURE_TYPES["surface"]
    
    def extract(self, image: np.ndarray) -> Dict:
        """Trích xuất đặc trưng bề mặt sử dụng texture analysis"""
        gray = self._to_grayscale(image)
        binary, contours = self._segment_pollen(image)
        
        if not contours:
            return {"surface_class": "unknown", "metrics": {}}
        
        # Tạo mask từ contour chính
        main_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [main_contour], -1, 255, -1)
        
        # Áp dụng mask để chỉ phân tích vùng phấn hoa
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Tính GLCM (Gray-Level Co-occurrence Matrix) features
        glcm_features = self._compute_glcm_features(masked_gray, mask)
        
        # Tính LBP (Local Binary Pattern) features
        lbp_features = self._compute_lbp_features(masked_gray, mask)
        
        # Phân loại bề mặt
        surface_class, confidence = self._classify_surface(glcm_features, lbp_features)
        
        return {
            "surface_class": surface_class,
            "confidence": confidence
        }

    def _compute_glcm_features(self, gray: np.ndarray, mask: np.ndarray) -> Dict:
        """Tính các đặc trưng từ GLCM"""
        masked_pixels = gray[mask > 0]
        
        if len(masked_pixels) == 0:
            return {"contrast": 0, "energy": 0, "homogeneity": 0, "correlation": 0}
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        contrast = np.std(masked_pixels)
        energy = np.sum(masked_pixels ** 2) / len(masked_pixels)
        homogeneity = 1 / (1 + np.var(masked_pixels))
        
        gradient_masked = gradient_magnitude[mask > 0]
        roughness = np.mean(gradient_masked) if len(gradient_masked) > 0 else 0
        
        return {
            "contrast": float(contrast),
            "energy": float(energy),
            "homogeneity": float(homogeneity),
            "roughness": float(roughness),
        }
    
    def _compute_lbp_features(self, gray: np.ndarray, mask: np.ndarray) -> Dict:
        """Tính Local Binary Pattern features"""
        lbp = np.zeros_like(gray)
        
        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] > center) << 7
                code |= (gray[i-1, j] > center) << 6
                code |= (gray[i-1, j+1] > center) << 5
                code |= (gray[i, j+1] > center) << 4
                code |= (gray[i+1, j+1] > center) << 3
                code |= (gray[i+1, j] > center) << 2
                code |= (gray[i+1, j-1] > center) << 1
                code |= (gray[i, j-1] > center) << 0
                lbp[i, j] = code
        
        lbp_masked = lbp[mask > 0]
        if len(lbp_masked) > 0:
            hist, _ = np.histogram(lbp_masked, bins=256, range=(0, 256))
            hist = hist.astype(float) / hist.sum()
            
            lbp_entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            lbp_uniformity = np.sum(hist ** 2)
        else:
            lbp_entropy = 0
            lbp_uniformity = 0
        
        return {
            "lbp_entropy": float(lbp_entropy),
            "lbp_uniformity": float(lbp_uniformity),
        }
    
    def _classify_surface(self, glcm: Dict, lbp: Dict) -> Tuple[str, float]:
        """Phân loại bề mặt dựa trên texture features"""
        roughness = glcm.get("roughness", 0)
        contrast = glcm.get("contrast", 0)
        lbp_entropy = lbp.get("lbp_entropy", 0)
        
        if roughness < 10 and contrast < 20:
            return "psilate", 0.8
        elif roughness < 20 and contrast < 40:
            return "scabrate", 0.7
        elif roughness > 50 and lbp_entropy > 6:
            return "echinate", 0.75
        elif lbp_entropy > 5 and roughness > 30:
            return "reticulate", 0.7
        elif contrast > 50:
            return "verrucate", 0.65
        else:
            return "striate", 0.6





class ApertureExtractor(BaseFeatureExtractor):
    """Trích xuất đặc trưng lỗ mở (aperture) của phấn hoa"""
    
    def __init__(self):
        super().__init__()
        self.feature_name = "aperture_type"
        self.aperture_classes = FEATURE_TYPES["aperture_type"]
    
    def extract(self, image: np.ndarray) -> Dict:
        """Trích xuất đặc trưng aperture"""
        gray = self._to_grayscale(image)
        binary, contours = self._segment_pollen(image)
        
        if not contours:
            return {"aperture_class": "unknown", "metrics": {}}
        
        main_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [main_contour], -1, 255, -1)
        
        # Detect apertures (holes/openings) using edge detection
        edges = cv2.Canny(gray, 50, 150)
        edges_masked = cv2.bitwise_and(edges, edges, mask=mask)
        
        # Find potential aperture regions
        aperture_contours, _ = cv2.findContours(edges_masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        significant_apertures = [c for c in aperture_contours 
                                 if cv2.contourArea(c) > 50]
        
        num_apertures = len(significant_apertures)
        
        # Phân tích đặc điểm của các aperture
        aperture_types = self._analyze_apertures(significant_apertures, gray.shape)
        
        # Phân loại
        aperture_class = self._classify_aperture(num_apertures, aperture_types)
        
        return {
            "aperture_class": aperture_class,
            "metrics": {
                "num_apertures": num_apertures,
                "aperture_types": aperture_types,
            }
        }
    
    def _analyze_apertures(self, contours: List, img_shape: Tuple) -> Dict:
        """Phân tích đặc điểm của các aperture"""
        if not contours:
            return {"colpi": 0, "pores": 0}
        
        colpi = 0  # Rãnh (elongated)
        pores = 0  # Lỗ (circular)
        
        for c in contours:
            if len(c) < 5:
                continue
            
            # Fit ellipse để xác định hình dạng
            try:
                ellipse = cv2.fitEllipse(c)
                (cx, cy), (major, minor), angle = ellipse
                
                if major > 0 and minor > 0:
                    ratio = major / minor
                    if ratio > 2:  # Elongated = colpus
                        colpi += 1
                    else:  # More circular = pore
                        pores += 1
            except:
                pores += 1
        
        return {"colpi": colpi, "pores": pores}
    
    def _classify_aperture(self, num_apertures: int, aperture_types: Dict) -> str:
        """Phân loại loại aperture"""
        colpi = aperture_types.get("colpi", 0)
        pores = aperture_types.get("pores", 0)
        
        if num_apertures == 0:
            return "inaperturate"
        elif colpi == 1 and pores == 0:
            return "monocolpate"
        elif colpi == 3 and pores == 0:
            return "tricolpate"
        elif colpi == 3 and pores > 0:
            return "tricolporate"
        elif pores == 3 and colpi == 0:
            return "triporate"
        elif pores > 3:
            return "pantoporate"
        else:
            return "tricolpate"  # Default


class ExineExtractor(BaseFeatureExtractor):
    """Trích xuất đặc trưng lớp vỏ ngoài (exine) của phấn hoa"""
    
    def __init__(self):
        super().__init__()
        self.feature_name = "exine"
        self.exine_classes = FEATURE_TYPES["exine"]
    
    def extract(self, image: np.ndarray) -> Dict:
        """Trích xuất đặc trưng exine"""
        gray = self._to_grayscale(image)
        binary, contours = self._segment_pollen(image)
        
        if not contours:
            return {"exine_class": "unknown", "metrics": {}}
        
        main_contour = max(contours, key=cv2.contourArea)
        
        # Tính độ dày exine bằng cách phân tích gradient tại biên
        exine_thickness = self._estimate_exine_thickness(gray, main_contour)
        
        # Phân tích cấu trúc phân tầng
        stratification = self._detect_stratification(gray, main_contour)
        
        # Phân loại
        exine_class = self._classify_exine(exine_thickness, stratification)
        
        return {
            "exine_class": exine_class,
            "metrics": {
                "thickness_pixels": float(exine_thickness),
            }
        }
    
    def _estimate_exine_thickness(self, gray: np.ndarray, contour: np.ndarray) -> float:
        """Ước tính độ dày exine từ gradient tại biên"""
        # Tạo mask từ contour
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Erode để lấy vùng bên trong
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        eroded = cv2.erode(mask, kernel, iterations=1)
        
        # Vùng exine = mask - eroded
        exine_region = cv2.subtract(mask, eroded)
        
        # Tính gradient trong vùng exine
        gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        exine_gradient = np.abs(gradient[exine_region > 0])
        
        # Độ dày tương đối dựa trên gradient
        thickness = np.mean(exine_gradient) if len(exine_gradient) > 0 else 0
        
        return thickness
    
    def _detect_stratification(self, gray: np.ndarray, contour: np.ndarray) -> float:
        """Phát hiện cấu trúc phân tầng trong exine"""
        # Tạo profile dọc theo pháp tuyến của contour
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Phân tích gradient theo hướng radial
        # Đơn giản: đếm số đỉnh trong histogram
        ring_mask = mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        layers = 0
        for i in range(3):
            eroded = cv2.erode(ring_mask, kernel, iterations=1)
            ring = cv2.subtract(ring_mask, eroded)
            
            ring_pixels = gray[ring > 0]
            if len(ring_pixels) > 10:
                # Kiểm tra sự thay đổi intensity
                if np.std(ring_pixels) > 20:
                    layers += 1
            
            ring_mask = eroded
        
        return layers / 3.0  # Normalize
    
    def _classify_exine(self, thickness: float, stratification: float) -> str:
        """Phân loại exine"""
        if thickness < 10:
            return "thin"
        elif thickness < 30:
            return "medium"
        elif stratification > 0.5:
            return "stratified"
        else:
            return "thick"


class SectionExtractor(BaseFeatureExtractor):
    """Xác định góc nhìn/mặt cắt của ảnh phấn hoa"""
    
    def __init__(self):
        super().__init__()
        self.feature_name = "section"
        self.section_classes = FEATURE_TYPES["section"]
    
    def extract(self, image: np.ndarray) -> Dict:
        """Xác định mặt cắt của ảnh phấn hoa"""
        gray = self._to_grayscale(image)
        binary, contours = self._segment_pollen(image)
        
        if not contours:
            return {"section_class": "unknown", "metrics": {}}
        
        main_contour = max(contours, key=cv2.contourArea)
        
        # Phân tích hình dạng để xác định góc nhìn
        if len(main_contour) >= 5:
            ellipse = cv2.fitEllipse(main_contour)
            (cx, cy), (major, minor), angle = ellipse
            aspect_ratio = major / minor if minor > 0 else 1
        else:
            aspect_ratio = 1
            angle = 0
        
        # Phân tích đối xứng
        symmetry_score = self._analyze_symmetry(gray, main_contour)
        
        # Phân loại mặt cắt
        section_class = self._classify_section(aspect_ratio, symmetry_score, angle)
        
        return {
            "section_class": section_class
        }
    
    def _analyze_symmetry(self, gray: np.ndarray, contour: np.ndarray) -> float:
        """Phân tích độ đối xứng của hạt phấn"""
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Tính centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # So sánh nửa trái và nửa phải
        left_half = mask[:, :cx]
        right_half = mask[:, cx:]
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Resize để cùng kích thước
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, :min_width]
        right_half_flipped = right_half_flipped[:, :min_width]
        
        # Tính độ tương đồng
        if left_half.size > 0 and right_half_flipped.size > 0:
            intersection = np.sum(np.logical_and(left_half > 0, right_half_flipped > 0))
            union = np.sum(np.logical_or(left_half > 0, right_half_flipped > 0))
            symmetry = intersection / union if union > 0 else 0
        else:
            symmetry = 0
        
        return symmetry
    
    def _classify_section(self, aspect_ratio: float, symmetry: float, angle: float) -> str:
        """Phân loại mặt cắt"""
        if aspect_ratio < 1.2 and symmetry > 0.8:
            return "polar"  # Nhìn từ cực, thường tròn và đối xứng
        elif aspect_ratio > 1.5 and symmetry > 0.6:
            return "equatorial"  # Nhìn từ xích đạo, thường elip
        else:
            return "oblique"  # Góc nhìn nghiêng


if __name__ == "__main__":
    # Test code
    extractor = PollenFeatureExtractor()
    
    # Tạo ảnh test
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.circle(test_image, (112, 112), 80, (100, 150, 200), -1)
    
    # Trích xuất đặc trưng
    features = extractor.extract_all_features(test_image)
    
    print("Extracted features:")
    for key, value in features.items():
        if key != "deep_features":
            print(f"  {key}: {value}")
        else:
            print(f"  deep_features: shape={value.shape}")
