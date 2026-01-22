"""
result_analyzer.py - Phân tích và bình luận kết quả trích xuất

Module này:
- Phân tích phân bố các đặc trưng
- Tạo biểu đồ thống kê
- Tạo báo cáo bình luận tự động
- So sánh với dữ liệu tham khảo
"""

import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


class ResultAnalyzer:
    """Phân tích và bình luận kết quả trích xuất đặc trưng"""
    
    def __init__(self, results_file: str):
        """
        Args:
            results_file: Đường dẫn file JSON chứa kết quả extraction
        """
        self.results_file = results_file
        self.results = self._load_results()
        self.df = self._parse_to_dataframe()
        self.analysis = {}
    
    def _load_results(self) -> List[Dict]:
        """Đọc file kết quả"""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _parse_to_dataframe(self) -> pd.DataFrame:
        """Chuyển kết quả thành DataFrame để phân tích"""
        rows = []
        
        for item in self.results:
            row = {
                'image_path': item.get('image_path', ''),
                'folder': os.path.basename(os.path.dirname(item.get('image_path', ''))),
            }
            
            # Extract features
            for feature in ['shape', 'surface', 'aperture_type', 'exine', 'section']:
                feat_dict = item.get(feature, {})
                if isinstance(feat_dict, dict):
                    row[f'{feature}_class'] = feat_dict.get(f'{feature}_class', 
                                                            feat_dict.get('class', None))
                    row[f'{feature}_confidence'] = feat_dict.get('confidence', None)
                else:
                    row[f'{feature}_class'] = feat_dict
            
            # Size
            size_dict = item.get('size', {})
            if isinstance(size_dict, dict):
                row['size_class'] = size_dict.get('size_class', None)
                row['size_value'] = size_dict.get('size_value', None)
                metrics = size_dict.get('metrics', {})
                row['diameter_micron'] = metrics.get('diameter_micron', None)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def analyze_all(self):
        """Thực hiện phân tích toàn diện"""
        print("\n" + "="*60)
        print("PHÂN TÍCH KẾT QUẢ TRÍCH XUẤT ĐẶC TRƯNG")
        print("="*60)
        
        self.analysis['general'] = self.analyze_general_statistics()
        self.analysis['shape'] = self.analyze_feature_distribution('shape')
        self.analysis['size'] = self.analyze_size_distribution()
        self.analysis['surface'] = self.analyze_feature_distribution('surface')
        self.analysis['aperture'] = self.analyze_feature_distribution('aperture_type')
        self.analysis['exine'] = self.analyze_feature_distribution('exine')
        self.analysis['section'] = self.analyze_feature_distribution('section')
        
        return self.analysis
    
    def analyze_general_statistics(self) -> Dict:
        """Phân tích thống kê chung"""
        print("\n📊 THỐNG KÊ TỔNG QUAN:")
        
        total_images = len(self.df)
        unique_folders = self.df['folder'].nunique()
        
        stats = {
            'total_images': total_images,
            'unique_folders': unique_folders,
            'images_per_folder': self.df['folder'].value_counts().to_dict()
        }
        
        print(f"  Tổng số ảnh: {total_images}")
        print(f"  Số folder: {unique_folders}")
        
        return stats
    
    def analyze_feature_distribution(self, feature: str) -> Dict:
        """Phân tích phân bố của một đặc trưng"""
        col_name = f'{feature}_class'
        
        if col_name not in self.df.columns:
            return {'error': f'Feature {feature} not found'}
        
        print(f"\n📈 PHÂN BỐ {feature.upper()}:")
        
        # Đếm số lượng mỗi class (loại bỏ NaN)
        distribution = self.df[col_name].dropna().value_counts()
        
        if len(distribution) == 0:
            print(f"  ⚠️  Không có dữ liệu")
            return {'error': 'No data available'}
        
        percentages = self.df[col_name].value_counts(normalize=True) * 100
        
        # In ra
        for cls, count in distribution.items():
            pct = percentages[cls]
            print(f"  {cls}: {count} ({pct:.1f}%)")
        
        # Phân tích
        most_common = distribution.index[0]
        least_common = distribution.index[-1]
        
        # Độ đa dạng (entropy)
        probs = distribution / distribution.sum()
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(distribution))
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0
        
        analysis = {
            'distribution': distribution.to_dict(),
            'percentages': percentages.to_dict(),
            'most_common': most_common,
            'least_common': least_common,
            'diversity_score': diversity_score,
            'num_classes': len(distribution)
        }
        
        print(f"  ➤ Phổ biến nhất: {most_common} ({distribution[most_common]} ảnh)")
        print(f"  ➤ Ít gặp nhất: {least_common} ({distribution[least_common]} ảnh)")
        print(f"  ➤ Độ đa dạng: {diversity_score:.2f} (0=đơn điệu, 1=đồng đều)")
        
        # Confidence analysis
        conf_col = f'{feature}_confidence'
        if conf_col in self.df.columns:
            avg_conf = self.df[conf_col].mean()
            if not pd.isna(avg_conf):
                print(f"  ➤ Độ tin cậy trung bình: {avg_conf:.2%}")
                analysis['avg_confidence'] = avg_conf
        
        return analysis
    
    def analyze_size_distribution(self) -> Dict:
        """Phân tích phân bố kích thước"""
        print(f"\n📏 PHÂN BỐ KÍCH THƯỚC:")
        
        if 'diameter_micron' not in self.df.columns:
            return {'error': 'Size data not found'}
        
        sizes = self.df['diameter_micron'].dropna()
        
        if len(sizes) == 0:
            return {'error': 'No valid size data'}
        
        stats = {
            'mean': sizes.mean(),
            'median': sizes.median(),
            'std': sizes.std(),
            'min': sizes.min(),
            'max': sizes.max(),
            'q25': sizes.quantile(0.25),
            'q75': sizes.quantile(0.75)
        }
        
        print(f"  Trung bình: {stats['mean']:.2f} μm")
        print(f"  Trung vị: {stats['median']:.2f} μm")
        print(f"  Độ lệch chuẩn: {stats['std']:.2f} μm")
        print(f"  Khoảng: {stats['min']:.2f} - {stats['max']:.2f} μm")
        print(f"  Tứ phân vị: {stats['q25']:.2f} - {stats['q75']:.2f} μm")
        
        # Size class distribution
        if 'size_class' in self.df.columns:
            size_class_dist = self.df['size_class'].value_counts()
            print(f"\n  Phân bố theo nhóm kích thước:")
            for cls, count in size_class_dist.items():
                pct = (count / len(self.df)) * 100
                print(f"    {cls}: {count} ({pct:.1f}%)")
            stats['class_distribution'] = size_class_dist.to_dict()
        
        return stats
    
    def generate_comments(self) -> str:
        """Tạo bình luận tự động về kết quả"""
        comments = []
        
        comments.append("# BÌNH LUẬN VÀ ĐÁNH GIÁ KẾT QUẢ\n")
        comments.append("## 1. Tổng quan\n")
        
        general = self.analysis.get('general', {})
        comments.append(f"Hệ thống đã xử lý thành công **{general.get('total_images', 0)}** ảnh phấn hoa ")
        comments.append(f"từ **{general.get('unique_folders', 0)}** folder khác nhau.\n")
        
        comments.append("\n## 2. Phân tích từng đặc trưng\n")
        
        # Shape
        if 'shape' in self.analysis:
            shape = self.analysis['shape']
            comments.append(f"\n### 2.1 Hình dạng (Shape)\n")
            comments.append(f"- Hình dạng phổ biến nhất: **{shape.get('most_common', 'N/A')}** ")
            comments.append(f"({shape.get('distribution', {}).get(shape.get('most_common', ''), 0)} ảnh)\n")
            
            diversity = shape.get('diversity_score', 0)
            if diversity > 0.7:
                comments.append(f"- Độ đa dạng hình dạng cao ({diversity:.2f}), cho thấy mẫu phấn có nhiều dạng khác nhau.\n")
            elif diversity < 0.3:
                comments.append(f"- Độ đa dạng hình dạng thấp ({diversity:.2f}), hầu hết mẫu có hình dạng tương tự.\n")
        
        # Size
        if 'size' in self.analysis:
            size = self.analysis['size']
            comments.append(f"\n### 2.2 Kích thước (Size)\n")
            comments.append(f"- Kích thước trung bình: **{size.get('mean', 0):.2f} μm** ")
            comments.append(f"(độ lệch chuẩn: {size.get('std', 0):.2f} μm)\n")
            comments.append(f"- Khoảng kích thước: {size.get('min', 0):.2f} - {size.get('max', 0):.2f} μm\n")
            
            mean_size = size.get('mean', 0)
            if mean_size < 25:
                comments.append("- Phấn hoa thuộc nhóm kích thước nhỏ (<25 μm)\n")
            elif mean_size < 50:
                comments.append("- Phấn hoa thuộc nhóm kích thước trung bình (25-50 μm)\n")
            else:
                comments.append("- Phấn hoa thuộc nhóm kích thước lớn (>50 μm)\n")
            
            cv = (size.get('std', 0) / mean_size) * 100 if mean_size > 0 else 0
            if cv > 20:
                comments.append(f"- Hệ số biến thiên cao ({cv:.1f}%), kích thước không đồng nhất\n")
            elif cv < 10:
                comments.append(f"- Hệ số biến thiên thấp ({cv:.1f}%), kích thước khá đồng nhất\n")
        
        # Surface
        if 'surface' in self.analysis:
            surface = self.analysis['surface']
            comments.append(f"\n### 2.3 Bề mặt (Surface)\n")
            comments.append(f"- Loại bề mặt phổ biến: **{surface.get('most_common', 'N/A')}**\n")
            
            avg_conf = surface.get('avg_confidence')
            if avg_conf:
                if avg_conf > 0.8:
                    comments.append(f"- Độ tin cậy cao ({avg_conf:.2%}), phân loại bề mặt khá chắc chắn\n")
                elif avg_conf < 0.6:
                    comments.append(f"- Độ tin cậy thấp ({avg_conf:.2%}), cần kiểm tra lại phương pháp texture analysis\n")
        
        # Aperture
        if 'aperture' in self.analysis and 'error' not in self.analysis['aperture']:
            aperture = self.analysis['aperture']
            comments.append(f"\n### 2.4 Lỗ mở (Aperture)\n")
            comments.append(f"- Loại aperture phổ biến: **{aperture.get('most_common', 'N/A')}**\n")
            
            num_classes = aperture.get('num_classes', 0)
            if num_classes == 1:
                comments.append("- Tất cả mẫu có cùng kiểu aperture, cho thấy đồng nhất về loài\n")
            elif num_classes > 3:
                comments.append(f"- Có {num_classes} kiểu aperture khác nhau, mẫu phấn đa dạng\n")
        else:
            comments.append(f"\n### 2.4 Lỗ mở (Aperture)\n")
            comments.append("- ⚠️ Không có dữ liệu aperture trong kết quả\n")
        
        comments.append("\n## 3. Kết luận\n")
        comments.append("\nDựa trên kết quả trích xuất đặc trưng:\n")
        
        # Tính chất tổng quát
        shape_div = self.analysis.get('shape', {}).get('diversity_score', 0)
        if shape_div > 0.6:
            comments.append("- Mẫu phấn hoa có độ đa dạng cao về hình thái\n")
        else:
            comments.append("- Mẫu phấn hoa tương đối đồng nhất về hình thái\n")
        
        # Chất lượng dữ liệu
        total = general.get('total_images', 1)
        if total > 100:
            comments.append("- Kích thước dataset đủ lớn để phân tích thống kê đáng tin cậy\n")
        elif total < 30:
            comments.append("- Dataset nhỏ, cần thu thập thêm dữ liệu để kết luận chắc chắn hơn\n")
        
        # Khuyến nghị
        comments.append("\n## 4. Khuyến nghị\n")
        
        if 'surface' in self.analysis:
            avg_conf = self.analysis['surface'].get('avg_confidence', 0)
            if avg_conf and avg_conf < 0.7:
                comments.append("- Cải thiện phương pháp texture analysis để tăng độ tin cậy phân loại bề mặt\n")
        
        if 'size' in self.analysis:
            cv = (self.analysis['size'].get('std', 0) / self.analysis['size'].get('mean', 1)) * 100
            if cv > 25:
                comments.append("- Kiểm tra lại quy trình đo kích thước do biến thiên cao\n")
        
        comments.append("- Nên thực hiện đánh giá định lượng với ground truth để xác minh độ chính xác\n")
        comments.append("- Xem xét phân tích theo từng folder/loài để có kết luận chi tiết hơn\n")
        
        return "".join(comments)
    
    def create_visualizations(self, output_dir: str = 'output'):
        """Tạo các biểu đồ trực quan"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📊 Đang tạo biểu đồ...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Shape distribution
        if 'shape_class' in self.df.columns:
            plt.figure(figsize=(10, 6))
            self.df['shape_class'].value_counts().plot(kind='bar', color='steelblue')
            plt.title('Phân bố Hình dạng (Shape)', fontsize=14, fontweight='bold')
            plt.xlabel('Loại hình dạng', fontsize=12)
            plt.ylabel('Số lượng', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'shape_distribution.png'), dpi=300)
            plt.close()
            print(f"  ✓ Đã lưu: shape_distribution.png")
        
        # 2. Size distribution (histogram)
        if 'diameter_micron' in self.df.columns:
            plt.figure(figsize=(10, 6))
            sizes = self.df['diameter_micron'].dropna()
            plt.hist(sizes, bins=30, color='coral', edgecolor='black', alpha=0.7)
            plt.axvline(sizes.mean(), color='red', linestyle='--', linewidth=2, label=f'Trung bình: {sizes.mean():.1f} μm')
            plt.axvline(sizes.median(), color='green', linestyle='--', linewidth=2, label=f'Trung vị: {sizes.median():.1f} μm')
            plt.title('Phân bố Kích thước (Size)', fontsize=14, fontweight='bold')
            plt.xlabel('Đường kính (μm)', fontsize=12)
            plt.ylabel('Tần số', fontsize=12)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'size_distribution.png'), dpi=300)
            plt.close()
            print(f"  ✓ Đã lưu: size_distribution.png")
        
        # 3. Surface distribution
        if 'surface_class' in self.df.columns:
            plt.figure(figsize=(10, 6))
            self.df['surface_class'].value_counts().plot(kind='barh', color='forestgreen')
            plt.title('Phân bố Bề mặt (Surface)', fontsize=14, fontweight='bold')
            plt.xlabel('Số lượng', fontsize=12)
            plt.ylabel('Loại bề mặt', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'surface_distribution.png'), dpi=300)
            plt.close()
            print(f"  ✓ Đã lưu: surface_distribution.png")
        
        # 4. Multi-feature comparison (pie charts)
        features = ['aperture_type', 'exine', 'section']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, feature in enumerate(features):
            col = f'{feature}_class'
            if col in self.df.columns:
                data = self.df[col].value_counts()
                axes[idx].pie(data, labels=data.index, autopct='%1.1f%%', startangle=90)
                axes[idx].set_title(feature.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'multi_feature_pie.png'), dpi=300)
        plt.close()
        print(f"  ✓ Đã lưu: multi_feature_pie.png")
        
        # 5. Size by class (boxplot)
        if 'diameter_micron' in self.df.columns and 'shape_class' in self.df.columns:
            plt.figure(figsize=(12, 6))
            self.df.boxplot(column='diameter_micron', by='shape_class', ax=plt.gca())
            plt.title('Kích thước theo Hình dạng', fontsize=14, fontweight='bold')
            plt.suptitle('')  # Remove auto title
            plt.xlabel('Loại hình dạng', fontsize=12)
            plt.ylabel('Đường kính (μm)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'size_by_shape.png'), dpi=300)
            plt.close()
            print(f"  ✓ Đã lưu: size_by_shape.png")
        
        print(f"\n✅ Đã tạo tất cả biểu đồ tại: {output_dir}/")
    
    def export_analysis_report(self, output_file: str = 'output/analysis_report.md'):
        """Xuất báo cáo phân tích ra file Markdown"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        comments = self.generate_comments()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(comments)
        
        print(f"\n✅ Đã lưu báo cáo phân tích tại: {output_file}")


def main():
    """Hàm main"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Phân tích và bình luận kết quả trích xuất',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ví dụ:
  python eval/result_analyzer.py --input output/extraction_results.json --output output/analysis_report.md
        '''
    )
    
    parser.add_argument('--input', '-i', required=True, 
                       help='File JSON chứa kết quả extraction')
    parser.add_argument('--output', '-o', default='output/analysis_report.md',
                       help='File Markdown output cho báo cáo')
    parser.add_argument('--plot-dir', default='output',
                       help='Thư mục lưu các biểu đồ')
    parser.add_argument('--no-plots', action='store_true',
                       help='Không tạo biểu đồ')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PHÂN TÍCH VÀ BÌNH LUẬN KẾT QUẢ")
    print("="*60)
    
    # Phân tích
    analyzer = ResultAnalyzer(args.input)
    analyzer.analyze_all()
    
    # Tạo biểu đồ
    if not args.no_plots:
        analyzer.create_visualizations(args.plot_dir)
    
    # Xuất báo cáo
    analyzer.export_analysis_report(args.output)
    
    print("\n" + "="*60)
    print("HOÀN THÀNH")
    print("="*60)


if __name__ == '__main__':
    main()
