# -*- coding: utf-8 -*-
"""
Chuyển đổi file JSON kết quả sang Excel
Tổng hợp theo từng thư mục con (0, 1, 2, 3, ...)
"""

import json
import pandas as pd
import os
from datetime import datetime


def get_folder_name(image_path: str) -> str:
    """Lấy tên thư mục cha của ảnh (0, 1, 2, 3, ...)"""
    parent = os.path.dirname(image_path)
    folder_name = os.path.basename(parent)
    return folder_name


def json_to_excel(json_path: str, excel_path: str = None):
    """
    Chuyển đổi file JSON extraction_results sang Excel
    Tổng hợp theo từng thư mục con
    
    Args:
        json_path: Đường dẫn file JSON
        excel_path: Đường dẫn file Excel đầu ra (tự động nếu None)
    """
    # Đọc file JSON
    print(f"Đang đọc file JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Tổng số ảnh: {len(data)}")
    
    # Chuyển đổi thành dạng phẳng để dễ đọc trong Excel
    rows = []
    for item in data:
        image_path = item.get('image_path', '')
        row = {
            # Thông tin ảnh
            'folder': get_folder_name(image_path),
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
            
            # Shape
            'shape_class': item.get('shape', {}).get('shape_class', ''),
            'shape_confidence': item.get('shape', {}).get('confidence', 0),
            'shape_area': item.get('shape', {}).get('metrics', {}).get('area', 0),
            'shape_perimeter': item.get('shape', {}).get('metrics', {}).get('perimeter', 0),
            
            # Size
            'size_class': item.get('size', {}).get('size_class', ''),
            'size_value': item.get('size', {}).get('size_value', ''),
            'size_full': item.get('size', {}).get('size_full', ''),
            'diameter_micron': item.get('size', {}).get('metrics', {}).get('diameter_micron', 0),
            'width_micron': item.get('size', {}).get('metrics', {}).get('width_micron', 0),
            'height_micron': item.get('size', {}).get('metrics', {}).get('height_micron', 0),
            
            # Surface
            'surface_class': item.get('surface', {}).get('surface_class', ''),
            'surface_confidence': item.get('surface', {}).get('confidence', 0),
            
            # Aperture
            'aperture_class': item.get('aperture_type', {}).get('aperture_class', ''),
            'num_apertures': item.get('aperture_type', {}).get('metrics', {}).get('num_apertures', 0),
            
            # Exine
            'exine_class': item.get('exine', {}).get('exine_class', ''),
            'exine_thickness': item.get('exine', {}).get('metrics', {}).get('thickness_pixels', 0),
            
            # Section
            'section_class': item.get('section', {}).get('section_class', ''),
        }
        rows.append(row)
    
    # Tạo DataFrame
    df = pd.DataFrame(rows)
    
    # Sắp xếp theo folder
    df['folder_num'] = pd.to_numeric(df['folder'], errors='coerce')
    df = df.sort_values(['folder_num', 'image_name']).drop('folder_num', axis=1)
    
    # Đường dẫn file Excel
    if excel_path is None:
        excel_path = json_path.replace('.json', '.xlsx')
    
    # Lưu ra Excel
    print(f"Đang lưu file Excel: {excel_path}")
    
    # Tạo Excel writer với formatting
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Dữ liệu chi tiết
        df.to_excel(writer, sheet_name='Chi tiết', index=False)
        
        # Sheet 2: Thống kê theo từng folder
        folders = sorted(df['folder'].unique(), key=lambda x: int(x) if x.isdigit() else 999)
        
        summary_rows = []
        for folder in folders:
            folder_df = df[df['folder'] == folder]
            summary_rows.append({
                'Folder': folder,
                'Số ảnh': len(folder_df),
                # Shape phổ biến nhất
                'Shape chính': folder_df['shape_class'].mode().iloc[0] if len(folder_df) > 0 else '',
                # Size phổ biến nhất
                'Size chính': folder_df['size_class'].mode().iloc[0] if len(folder_df) > 0 else '',
                'Size TB (μm)': round(folder_df['diameter_micron'].mean(), 2),
                # Surface phổ biến nhất
                'Surface chính': folder_df['surface_class'].mode().iloc[0] if len(folder_df) > 0 else '',
                # Aperture phổ biến nhất
                'Aperture chính': folder_df['aperture_class'].mode().iloc[0] if len(folder_df) > 0 else '',
                # Exine phổ biến nhất
                'Exine chính': folder_df['exine_class'].mode().iloc[0] if len(folder_df) > 0 else '',
                # Section phổ biến nhất
                'Section chính': folder_df['section_class'].mode().iloc[0] if len(folder_df) > 0 else '',
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name='Tổng hợp theo Folder', index=False)
        
        # Sheet 3: Thống kê chi tiết từng folder
        detail_stats = []
        for folder in folders:
            folder_df = df[df['folder'] == folder]
            
            # Đếm từng loại
            shape_counts = folder_df['shape_class'].value_counts().to_dict()
            size_counts = folder_df['size_class'].value_counts().to_dict()
            surface_counts = folder_df['surface_class'].value_counts().to_dict()
            aperture_counts = folder_df['aperture_class'].value_counts().to_dict()
            exine_counts = folder_df['exine_class'].value_counts().to_dict()
            section_counts = folder_df['section_class'].value_counts().to_dict()
            
            detail_stats.append({
                'Folder': folder,
                'Tổng ảnh': len(folder_df),
                # Shape
                'shape_spherical': shape_counts.get('spherical', 0),
                'shape_ellipsoidal': shape_counts.get('ellipsoidal', 0),
                'shape_triangular': shape_counts.get('triangular', 0),
                'shape_rectangular': shape_counts.get('rectangular', 0),
                'shape_irregular': shape_counts.get('irregular', 0),
                # Size
                'size_very_small': size_counts.get('very_small', 0),
                'size_small': size_counts.get('small', 0),
                'size_medium': size_counts.get('medium', 0),
                'size_large': size_counts.get('large', 0),
                'size_very_large': size_counts.get('very_large', 0),
                # Surface
                'surface_psilate': surface_counts.get('psilate', 0),
                'surface_scabrate': surface_counts.get('scabrate', 0),
                'surface_verrucate': surface_counts.get('verrucate', 0),
                'surface_echinate': surface_counts.get('echinate', 0),
                'surface_reticulate': surface_counts.get('reticulate', 0),
                'surface_striate': surface_counts.get('striate', 0),
                # Aperture
                'aperture_inaperturate': aperture_counts.get('inaperturate', 0),
                'aperture_monocolpate': aperture_counts.get('monocolpate', 0),
                'aperture_tricolpate': aperture_counts.get('tricolpate', 0),
                'aperture_tricolporate': aperture_counts.get('tricolporate', 0),
                'aperture_triporate': aperture_counts.get('triporate', 0),
                'aperture_pantoporate': aperture_counts.get('pantoporate', 0),
                # Exine
                'exine_thin': exine_counts.get('thin', 0),
                'exine_medium': exine_counts.get('medium', 0),
                'exine_thick': exine_counts.get('thick', 0),
                'exine_stratified': exine_counts.get('stratified', 0),
                # Section
                'section_polar': section_counts.get('polar', 0),
                'section_equatorial': section_counts.get('equatorial', 0),
                'section_oblique': section_counts.get('oblique', 0),
            })
        
        detail_df = pd.DataFrame(detail_stats)
        detail_df.to_excel(writer, sheet_name='Chi tiết theo Folder', index=False)
        
        # Sheet 4: Thống kê tổng
        total_stats = pd.DataFrame({
            'Đặc trưng': ['SHAPE', 'SIZE', 'SURFACE', 'APERTURE', 'EXINE', 'SECTION'],
            'Tổng số mẫu': [len(df)] * 6,
            'Số folder': [len(folders)] * 6,
        })
        total_stats.to_excel(writer, sheet_name='Tổng quan', index=False)
    
    print(f"\n✅ Đã tạo file Excel thành công!")
    print(f"   📊 File: {excel_path}")
    print(f"   📝 Số dòng: {len(df)}")
    print(f"   📁 Số folder: {len(folders)}")
    
    # In thống kê theo folder
    print(f"\n📈 THỐNG KÊ THEO FOLDER:")
    for folder in folders:
        folder_df = df[df['folder'] == folder]
        print(f"   Folder {folder}: {len(folder_df)} ảnh")
    
    return excel_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Chuyển JSON sang Excel')
    parser.add_argument('--input', '-i', type=str, 
                        default='output/extraction_results.json',
                        help='Đường dẫn file JSON')
    parser.add_argument('--output', '-o', type=str, 
                        default=None,
                        help='Đường dẫn file Excel đầu ra')
    
    args = parser.parse_args()
    
    json_to_excel(args.input, args.output)
