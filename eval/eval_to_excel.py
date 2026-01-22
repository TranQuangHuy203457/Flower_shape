"""
eval_to_excel.py - Xuất báo cáo đánh giá ra Excel chi tiết

Tạo file Excel với nhiều sheets:
- Tóm tắt: Metrics tổng quan
- Confusion Matrix: Ma trận nhầm lẫn cho mỗi đặc trưng
- Chi tiết: Bảng so sánh từng ảnh
- Phân tích: Biểu đồ và nhận xét
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from eval.evaluator import PollenEvaluator
import json
from typing import Dict


class ExcelReporter:
    """Xuất báo cáo đánh giá ra Excel với format đẹp"""
    
    def __init__(self, evaluator: PollenEvaluator):
        self.evaluator = evaluator
        self.results = evaluator.results
    
    def create_summary_sheet(self) -> pd.DataFrame:
        """Tạo sheet tóm tắt metrics"""
        data = []
        
        # Categorical features
        for feature in self.evaluator.categorical_features:
            if feature in self.results and 'accuracy' in self.results[feature]:
                r = self.results[feature]
                data.append({
                    'Đặc trưng': feature.upper(),
                    'Loại': 'Phân loại',
                    'Số mẫu': r['num_samples'],
                    'Accuracy': f"{r['accuracy']:.2%}",
                    'Precision (Macro)': f"{r['macro_precision']:.2%}",
                    'Recall (Macro)': f"{r['macro_recall']:.2%}",
                    'F1-Score (Macro)': f"{r['macro_f1']:.2%}",
                    'MAE': '-',
                    'RMSE': '-',
                    'R²': '-'
                })
        
        # Numeric features
        for feature in self.evaluator.numeric_features:
            if feature in self.results and 'mae' in self.results[feature]:
                r = self.results[feature]
                data.append({
                    'Đặc trưng': feature.upper(),
                    'Loại': 'Số',
                    'Số mẫu': r['num_samples'],
                    'Accuracy': '-',
                    'Precision (Macro)': '-',
                    'Recall (Macro)': '-',
                    'F1-Score (Macro)': '-',
                    'MAE': f"{r['mae']:.2f} μm",
                    'RMSE': f"{r['rmse']:.2f} μm",
                    'R²': f"{r['r2']:.3f}"
                })
        
        # Overall summary
        if 'summary' in self.results:
            s = self.results['summary']
            if 'overall_accuracy' in s:
                data.append({
                    'Đặc trưng': 'TRUNG BÌNH',
                    'Loại': 'Tổng quan',
                    'Số mẫu': s['total_samples'],
                    'Accuracy': f"{s['overall_accuracy']:.2%}",
                    'Precision (Macro)': '-',
                    'Recall (Macro)': '-',
                    'F1-Score (Macro)': f"{s['overall_macro_f1']:.2%}",
                    'MAE': '-',
                    'RMSE': '-',
                    'R²': '-'
                })
        
        return pd.DataFrame(data)
    
    def create_confusion_matrices(self) -> Dict[str, pd.DataFrame]:
        """Tạo confusion matrix cho mỗi đặc trưng"""
        matrices = {}
        
        for feature in self.evaluator.categorical_features:
            if feature in self.results and 'confusion_matrix' in self.results[feature]:
                cm = self.results[feature]['confusion_matrix']
                classes = cm['classes']
                matrix_data = cm['matrix']
                
                # Tạo DataFrame
                df = pd.DataFrame(
                    [[matrix_data[true_cls][pred_cls] for pred_cls in classes] 
                     for true_cls in classes],
                    index=classes,
                    columns=classes
                )
                
                # Thêm tổng hàng và cột
                df['Tổng'] = df.sum(axis=1)
                df.loc['Tổng'] = df.sum()
                
                matrices[feature] = df
        
        return matrices
    
    def create_per_class_detail(self) -> Dict[str, pd.DataFrame]:
        """Tạo bảng chi tiết từng class"""
        details = {}
        
        for feature in self.evaluator.categorical_features:
            if feature in self.results and 'per_class' in self.results[feature]:
                r = self.results[feature]
                per_class = r['per_class']
                precision = r['precision']
                recall = r['recall']
                f1 = r['f1']
                
                data = []
                for cls in sorted(per_class.keys()):
                    pc = per_class[cls]
                    data.append({
                        'Class': cls,
                        'Support (GT)': pc['support'],
                        'Predicted': pc['predicted'],
                        'True Positive': pc['tp'],
                        'False Positive': pc['fp'],
                        'False Negative': pc['fn'],
                        'Precision': f"{precision[cls]:.2%}",
                        'Recall': f"{recall[cls]:.2%}",
                        'F1-Score': f"{f1[cls]:.2%}"
                    })
                
                details[feature] = pd.DataFrame(data)
        
        return details
    
    def create_detailed_predictions(self) -> pd.DataFrame:
        """Tạo bảng chi tiết predictions từng ảnh"""
        return self.evaluator.export_to_dataframe()
    
    def create_analysis_sheet(self) -> pd.DataFrame:
        """Tạo sheet phân tích và nhận xét"""
        analyses = []
        
        # Phân tích từng đặc trưng
        for feature in self.evaluator.categorical_features:
            if feature in self.results and 'accuracy' in self.results[feature]:
                r = self.results[feature]
                
                # Xác định class tốt nhất và tệ nhất
                f1_scores = r['f1']
                if f1_scores:
                    sorted_f1 = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
                    best_class = sorted_f1[0]
                    worst_class = sorted_f1[-1]
                    
                    # Đánh giá hiệu năng
                    if r['accuracy'] >= 0.8:
                        performance = "Tốt"
                    elif r['accuracy'] >= 0.6:
                        performance = "Trung bình"
                    else:
                        performance = "Cần cải thiện"
                    
                    analyses.append({
                        'Đặc trưng': feature.upper(),
                        'Hiệu năng': performance,
                        'Accuracy': f"{r['accuracy']:.2%}",
                        'F1 (Macro)': f"{r['macro_f1']:.2%}",
                        'Class tốt nhất': f"{best_class[0]} (F1={best_class[1]:.2%})",
                        'Class tệ nhất': f"{worst_class[0]} (F1={worst_class[1]:.2%})",
                        'Nhận xét': self._generate_comment(feature, r)
                    })
        
        # Size analysis
        for feature in self.evaluator.numeric_features:
            if feature in self.results and 'mae' in self.results[feature]:
                r = self.results[feature]
                
                # Đánh giá dựa trên MAE
                mean_true = r['mean_true']
                relative_error = (r['mae'] / mean_true) * 100 if mean_true > 0 else 0
                
                if relative_error < 5:
                    performance = "Tốt"
                elif relative_error < 10:
                    performance = "Trung bình"
                else:
                    performance = "Cần cải thiện"
                
                analyses.append({
                    'Đặc trưng': feature.upper(),
                    'Hiệu năng': performance,
                    'Accuracy': '-',
                    'F1 (Macro)': '-',
                    'Class tốt nhất': f"MAE: {r['mae']:.2f} μm",
                    'Class tệ nhất': f"Lỗi tương đối: {relative_error:.1f}%",
                    'Nhận xét': self._generate_size_comment(r)
                })
        
        return pd.DataFrame(analyses)
    
    def _generate_comment(self, feature: str, results: Dict) -> str:
        """Tạo nhận xét tự động cho đặc trưng phân loại"""
        accuracy = results['accuracy']
        f1 = results['macro_f1']
        
        comments = []
        
        if accuracy < 0.6:
            comments.append("Hiệu năng thấp, cần cải thiện thuật toán hoặc chất lượng dữ liệu.")
        elif accuracy < 0.8:
            comments.append("Hiệu năng chấp nhận được nhưng vẫn có thể cải thiện.")
        else:
            comments.append("Hiệu năng tốt.")
        
        # Kiểm tra sự chênh lệch giữa các class
        f1_scores = list(results['f1'].values())
        if f1_scores:
            f1_std = np.std(f1_scores)
            if f1_std > 0.2:
                comments.append("Có sự chênh lệch lớn giữa các class, cần cân bằng dữ liệu.")
        
        # Kiểm tra confusion
        cm = results.get('confusion_matrix', {})
        if cm:
            # Tìm cặp class bị nhầm nhiều nhất
            matrix = cm['matrix']
            max_confusion = 0
            confused_pair = None
            
            for true_cls in matrix:
                for pred_cls in matrix[true_cls]:
                    if true_cls != pred_cls and matrix[true_cls][pred_cls] > max_confusion:
                        max_confusion = matrix[true_cls][pred_cls]
                        confused_pair = (true_cls, pred_cls)
            
            if confused_pair and max_confusion > 5:
                comments.append(f"Class '{confused_pair[0]}' thường bị nhầm với '{confused_pair[1]}' ({max_confusion} lần).")
        
        return " ".join(comments)
    
    def _generate_size_comment(self, results: Dict) -> str:
        """Tạo nhận xét cho đánh giá size"""
        mae = results['mae']
        bias = results['bias']
        r2 = results['r2']
        
        comments = []
        
        if mae < 2:
            comments.append("Độ chính xác đo kích thước rất tốt.")
        elif mae < 5:
            comments.append("Độ chính xác đo kích thước ở mức chấp nhận được.")
        else:
            comments.append("Sai số đo lường cao, cần kiểm tra lại tỷ lệ chuyển đổi pixel-micron.")
        
        if abs(bias) > 1:
            if bias > 0:
                comments.append(f"Có xu hướng đo lớn hơn thực tế (bias: +{bias:.2f} μm).")
            else:
                comments.append(f"Có xu hướng đo nhỏ hơn thực tế (bias: {bias:.2f} μm).")
        
        if r2 < 0.7:
            comments.append("Tương quan thấp, cần cải thiện phương pháp segmentation.")
        
        return " ".join(comments)
    
    def export_to_excel(self, output_file: str):
        """Xuất toàn bộ báo cáo ra Excel"""
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Sheet 1: Tóm tắt
                summary_df = self.create_summary_sheet()
                summary_df.to_excel(writer, sheet_name='Tóm tắt', index=False)
                
                # Sheet 2: Phân tích và nhận xét
                analysis_df = self.create_analysis_sheet()
                analysis_df.to_excel(writer, sheet_name='Phân tích', index=False)
                
                # Sheet 3: Chi tiết predictions
                detail_df = self.create_detailed_predictions()
                detail_df.to_excel(writer, sheet_name='Chi tiết', index=False)
                
                # Sheets 4+: Confusion matrices
                confusion_matrices = self.create_confusion_matrices()
                for feature, cm_df in confusion_matrices.items():
                    sheet_name = f'CM_{feature}'[:31]  # Excel sheet name limit
                    cm_df.to_excel(writer, sheet_name=sheet_name)
                
                # Sheets: Per-class details
                per_class_details = self.create_per_class_detail()
                for feature, pc_df in per_class_details.items():
                    sheet_name = f'Detail_{feature}'[:31]
                    pc_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"✅ Đã xuất báo cáo Excel tại: {output_file}")
            
        except Exception as e:
            print(f"⚠️  Không thể tạo file Excel: {e}")
            print("Đang xuất ra CSV thay thế...")
            
            # Fallback: export to CSV
            base_name = output_file.replace('.xlsx', '')
            
            summary_df = self.create_summary_sheet()
            summary_df.to_csv(f"{base_name}_summary.csv", index=False, encoding='utf-8-sig')
            
            detail_df = self.create_detailed_predictions()
            detail_df.to_csv(f"{base_name}_detail.csv", index=False, encoding='utf-8-sig')
            
            print(f"✅ Đã xuất CSV tại: {base_name}_*.csv")


def main():
    """Hàm main"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Xuất báo cáo đánh giá ra Excel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ví dụ:
  python eval/eval_to_excel.py --pred output/extraction_results.json --gt data/ground_truth.json --out output/evaluation.xlsx
        '''
    )
    
    parser.add_argument('--pred', required=True, help='File JSON predictions')
    parser.add_argument('--gt', required=True, help='File JSON ground truth')
    parser.add_argument('--out', '-o', default='output/evaluation.xlsx', 
                       help='File Excel output')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ĐÁNH GIÁ VÀ XUẤT BÁO CÁO EXCEL")
    print("="*60)
    
    # Chạy evaluation
    print("\n1. Đang thực hiện đánh giá...")
    evaluator = PollenEvaluator(args.pred, args.gt)
    evaluator.evaluate_all()
    evaluator.generate_report()
    
    # Xuất Excel
    print("\n2. Đang tạo báo cáo Excel...")
    reporter = ExcelReporter(evaluator)
    reporter.export_to_excel(args.out)
    
    print("\n" + "="*60)
    print("HOÀN THÀNH")
    print("="*60)


if __name__ == '__main__':
    main()
