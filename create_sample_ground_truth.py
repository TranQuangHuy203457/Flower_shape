"""
create_sample_ground_truth.py - Tạo ground truth mẫu để demo evaluation

Script này lấy một mẫu ngẫu nhiên từ extraction results và tạo ground truth
với một số điều chỉnh nhẹ để có thể đánh giá metrics.
"""

import json
import random
import os

def create_sample_ground_truth(extraction_file, output_file, num_samples=50):
    """Tạo ground truth mẫu từ extraction results"""
    
    print(f"Đọc file extraction results...")
    with open(extraction_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"Tổng số: {len(results)} kết quả")
    print(f"Chọn ngẫu nhiên {num_samples} mẫu...")
    
    # Chọn ngẫu nhiên
    sampled = random.sample(results, min(num_samples, len(results)))
    
    ground_truth = []
    
    for item in sampled:
        # Lấy image path
        image_path = item.get('image_path', '')
        image_name = os.path.basename(image_path)
        
        # Tạo ground truth entry
        gt_entry = {
            'image': image_name,
        }
        
        # Shape
        shape = item.get('shape', {})
        if isinstance(shape, dict):
            gt_entry['shape'] = shape.get('shape_class', 'unknown')
        
        # Size
        size = item.get('size', {})
        if isinstance(size, dict):
            metrics = size.get('metrics', {})
            size_um = metrics.get('diameter_micron', 0)
            # Thêm một chút variation (~5%) để có sai số
            variation = random.uniform(-0.05, 0.05)
            gt_entry['size_um'] = round(size_um * (1 + variation), 2)
        
        # Surface
        surface = item.get('surface', {})
        if isinstance(surface, dict):
            gt_entry['surface'] = surface.get('surface_class', 'unknown')
        
        # Aperture
        aperture = item.get('aperture_type', {})
        if isinstance(aperture, dict):
            gt_entry['aperture_type'] = aperture.get('aperture_class', 'unknown')
        
        # Exine
        exine = item.get('exine', {})
        if isinstance(exine, dict):
            gt_entry['exine'] = exine.get('exine_class', 'unknown')
        
        # Section
        section = item.get('section', {})
        if isinstance(section, dict):
            gt_entry['section'] = section.get('section_class', 'unknown')
        
        ground_truth.append(gt_entry)
    
    # Thêm một số sai lệch để có confusion (~10% error rate)
    num_errors = int(len(ground_truth) * 0.1)
    error_indices = random.sample(range(len(ground_truth)), num_errors)
    
    shape_classes = ['spherical', 'ellipsoidal', 'triangular', 'irregular']
    surface_classes = ['psilate', 'scabrate', 'echinate', 'reticulate', 'striate']
    
    for idx in error_indices:
        # Thay đổi ngẫu nhiên một field
        field = random.choice(['shape', 'surface'])
        if field == 'shape':
            current = ground_truth[idx].get('shape', 'unknown')
            others = [s for s in shape_classes if s != current]
            if others:
                ground_truth[idx]['shape'] = random.choice(others)
        elif field == 'surface':
            current = ground_truth[idx].get('surface', 'unknown')
            others = [s for s in surface_classes if s != current]
            if others:
                ground_truth[idx]['surface'] = random.choice(others)
    
    # Lưu file
    print(f"\nLưu ground truth vào {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Đã tạo {len(ground_truth)} ground truth entries")
    print(f"   - Có ~{num_errors} entries với sai lệch để test confusion matrix")
    
    # In mẫu
    print("\n📋 Mẫu ground truth:")
    for i, gt in enumerate(ground_truth[:3]):
        print(f"\n{i+1}. {gt.get('image', 'N/A')}")
        print(f"   Shape: {gt.get('shape', 'N/A')}")
        print(f"   Size: {gt.get('size_um', 0):.2f} μm")
        print(f"   Surface: {gt.get('surface', 'N/A')}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Tạo ground truth mẫu từ extraction results')
    parser.add_argument('--input', default='output/extraction_results.json',
                       help='File extraction results')
    parser.add_argument('--output', default='data/ground_truth.json',
                       help='File ground truth output')
    parser.add_argument('--num', type=int, default=50,
                       help='Số lượng mẫu')
    
    args = parser.parse_args()
    
    create_sample_ground_truth(args.input, args.output, args.num)
