import os
import argparse
import json
import gc
from typing import List, Dict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(f"{desc}...")
        for i, item in enumerate(iterable):
            if i % 50 == 0:
                print(f"  Đã xử lý {i} ảnh...")
            yield item

from pollen_features.feature_extractor import PollenFeatureExtractor
from pollen_features.database_handler import DatabaseHandler
from config import DATA_DIR, IMAGES_DIR, OUTPUT_DIR


def process_single_image(image_path: str, extractor: PollenFeatureExtractor, 
                         db_handler: DatabaseHandler = None, 
                         verbose: bool = True) -> Dict:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    
    # Trích xuất đặc trưng
    features = extractor.extract_all_features(image_path)
    features['image_path'] = image_path
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ẢNH: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        print("\n📐 HÌNH DẠNG (Shape):")
        shape = features.get('shape', {})
        print(f"   Phân loại: {shape.get('shape_class', 'N/A')}")
        print(f"   Độ tin cậy: {shape.get('confidence', 0):.2%}")
        if 'metrics' in shape:
            print(f"   - Độ tròn: {shape['metrics'].get('circularity', 0):.3f}")
            print(f"   - Tỷ lệ khung hình: {shape['metrics'].get('aspect_ratio', 0):.3f}")
        
        print("\n📏 KÍCH THƯỚC (Size):")
        size = features.get('size', {})
        print(f"   Phân loại: {size.get('size_class', 'N/A')}")
        if 'metrics' in size:
            print(f"   - Đường kính: {size['metrics'].get('diameter_micron', 0):.2f} μm")
        
        print("\n🔍 BỀ MẶT (Surface):")
        surface = features.get('surface', {})
        print(f"   Phân loại: {surface.get('surface_class', 'N/A')}")
        print(f"   Độ tin cậy: {surface.get('confidence', 0):.2%}")
        
        print("\n🕳️ LỖ MỞ (Aperture):")
        aperture = features.get('aperture_type', {})
        print(f"   Phân loại: {aperture.get('aperture_class', 'N/A')}")
        
        print("\n🧱 LỚP VỎ (Exine):")
        exine = features.get('exine', {})
        print(f"   Phân loại: {exine.get('exine_class', 'N/A')}")
        
        print("\n📷 MẶT CẮT (Section):")
        section = features.get('section', {})
        print(f"   Phân loại: {section.get('section_class', 'N/A')}")
    
    # Lưu vào database nếu có
    if db_handler:
        image_id = db_handler.add_image(image_path)
        db_handler.add_features(image_id, features)
    
    return features


def process_directory(dir_path: str, extractor: PollenFeatureExtractor,
                      db_handler: DatabaseHandler = None,
                      extensions: List[str] = None) -> List[Dict]:
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # Tìm tất cả ảnh trong thư mục
    image_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(root, file))
    
    print(f"Tìm thấy {len(image_files)} ảnh trong {dir_path}")
    
    results = []
    errors = []
    batch_size = 20  # Giải phóng bộ nhớ sau mỗi 20 ảnh
    
    for i, image_path in enumerate(tqdm(image_files, desc="Đang xử lý")):
        try:
            features = process_single_image(image_path, extractor, db_handler, verbose=False)
            results.append(features)
        except Exception as e:
            errors.append({'image_path': image_path, 'error': str(e)})
        
        # Giải phóng bộ nhớ định kỳ
        if (i + 1) % batch_size == 0:
            gc.collect()
    
    print(f"\nHoàn thành: {len(results)}/{len(image_files)} ảnh")
    if errors:
        print(f"Số lỗi: {len(errors)}")
    
    return results


def save_results_to_json(results: List[Dict], output_path: str):
    """Lưu kết quả ra file JSON"""
    # Chuyển đổi numpy arrays thành list
    def convert_for_json(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        return obj
    
    # Loại bỏ deep_features vì quá lớn
    results_clean = []
    for r in results:
        r_clean = {k: v for k, v in r.items() if k != 'deep_features'}
        results_clean.append(convert_for_json(r_clean))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_clean, f, ensure_ascii=False, indent=2)
    
    print(f"Đã lưu kết quả tại: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Trích xuất đặc trưng từ ảnh phấn hoa',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ví dụ sử dụng:
  # Xử lý một ảnh đơn lẻ
  python main.py --image path/to/pollen.jpg
  
  # Xử lý thư mục ảnh
  python main.py --dir path/to/images/
  
  # Xử lý và lưu vào database
  python main.py --dir path/to/images/ --save-db
  
  # Xuất kết quả ra JSON
  python main.py --dir path/to/images/ --output results.json
        '''
    )
    
    parser.add_argument('--image', '-i', type=str,
                        help='Đường dẫn đến ảnh cần xử lý')
    parser.add_argument('--dir', '-d', type=str,
                        help='Đường dẫn thư mục chứa ảnh')
    parser.add_argument('--output', '-o', type=str,
                        help='Đường dẫn file JSON để lưu kết quả')
    parser.add_argument('--save-db', action='store_true',
                        help='Lưu kết quả vào database SQLite')
    parser.add_argument('--model', '-m', type=str,
                        help='Đường dẫn đến model đã train (optional)')
    
    args = parser.parse_args()
    
    if not args.image and not args.dir:
        parser.print_help()
        print("\n⚠️  Vui lòng cung cấp --image hoặc --dir")
        return
    
    # Khởi tạo extractor
    print("Đang khởi tạo bộ trích xuất đặc trưng...")
    extractor = PollenFeatureExtractor(model_path=args.model)
    
    # Khởi tạo database handler nếu cần
    db_handler = DatabaseHandler() if args.save_db else None
    
    # Xử lý
    if args.image:
        result = process_single_image(args.image, extractor, db_handler)
        results = [result]
    else:
        results = process_directory(args.dir, extractor, db_handler)
    
    # Lưu kết quả
    if args.output:
        save_results_to_json(results, args.output)
    elif args.dir:
        # Tự động lưu ra output
        output_path = os.path.join(OUTPUT_DIR, 'extraction_results.json')
        save_results_to_json(results, output_path)
    
    # Đóng database
    if db_handler:
        db_handler.close()
        print(f"Đã lưu vào database")
    
    print("\n✅ Hoàn thành trích xuất đặc trưng!")


if __name__ == "__main__":
    main()
