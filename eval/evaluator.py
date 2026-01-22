"""
evaluator.py - Đánh giá định lượng kết quả trích xuất đặc trưng

Module này thực hiện:
- So sánh predictions với ground truth
- Tính accuracy, precision, recall, F1-score cho các đặc trưng phân loại
- Tính MAE, RMSE cho đặc trưng số (size)
- Tạo confusion matrix
- Xuất báo cáo chi tiết
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import os


class PollenEvaluator:
    """Đánh giá hiệu năng trích xuất đặc trưng phấn hoa"""
    
    def __init__(self, predictions_file: str, ground_truth_file: str):
        """
        Args:
            predictions_file: Đường dẫn file JSON chứa kết quả predictions
            ground_truth_file: Đường dẫn file JSON chứa ground truth
        """
        self.predictions = self._load_json(predictions_file)
        self.ground_truth = self._load_json(ground_truth_file)
        self.matched_data = self._match_predictions_with_gt()
        
        # Các loại đặc trưng cần đánh giá
        self.categorical_features = ['shape', 'surface', 'aperture_type', 'exine', 'section']
        self.numeric_features = ['size']
        
        self.results = {}
    
    def _load_json(self, filepath: str) -> List[Dict]:
        """Đọc file JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _match_predictions_with_gt(self) -> List[Dict]:
        """Khớp predictions với ground truth dựa trên image path"""
        gt_dict = {}
        for item in self.ground_truth:
            # Lấy basename để so khớp
            image_key = os.path.basename(item.get('image', item.get('image_path', '')))
            gt_dict[image_key] = item
        
        matched = []
        for pred in self.predictions:
            pred_image = os.path.basename(pred.get('image_path', ''))
            if pred_image in gt_dict:
                matched.append({
                    'prediction': pred,
                    'ground_truth': gt_dict[pred_image]
                })
        
        print(f"Matched {len(matched)}/{len(self.predictions)} predictions with ground truth")
        return matched
    
    def evaluate_all(self) -> Dict:
        """Thực hiện đánh giá toàn bộ"""
        print("\n" + "="*60)
        print("BẮT ĐẦU ĐÁNH GIÁ ĐỊNH LƯỢNG")
        print("="*60)
        
        # Đánh giá từng loại đặc trưng
        for feature in self.categorical_features:
            self.results[feature] = self.evaluate_categorical(feature)
        
        for feature in self.numeric_features:
            self.results[feature] = self.evaluate_numeric(feature)
        
        # Tính tổng quan
        self.results['summary'] = self.compute_summary()
        
        return self.results
    
    def evaluate_categorical(self, feature_name: str) -> Dict:
        """Đánh giá đặc trưng phân loại"""
        print(f"\n📊 Đánh giá {feature_name}...")
        
        y_true = []
        y_pred = []
        
        for item in self.matched_data:
            # Lấy giá trị ground truth
            gt = item['ground_truth'].get(feature_name, None)
            
            # Lấy giá trị prediction
            pred_dict = item['prediction'].get(feature_name, {})
            if isinstance(pred_dict, dict):
                pred = pred_dict.get(f'{feature_name}_class', 
                                     pred_dict.get('class', None))
            else:
                pred = pred_dict
            
            if gt and pred:
                y_true.append(gt)
                y_pred.append(pred)
        
        if not y_true:
            return {'error': 'No data to evaluate'}
        
        # Tính metrics
        accuracy = self._accuracy(y_true, y_pred)
        precision, recall, f1 = self._precision_recall_f1(y_true, y_pred)
        confusion = self._confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        per_class = self._per_class_metrics(y_true, y_pred)
        
        result = {
            'accuracy': accuracy,
            'macro_precision': np.mean(list(precision.values())),
            'macro_recall': np.mean(list(recall.values())),
            'macro_f1': np.mean(list(f1.values())),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': confusion,
            'per_class': per_class,
            'num_samples': len(y_true)
        }
        
        # In kết quả
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Macro F1: {result['macro_f1']:.2%}")
        print(f"  Samples: {len(y_true)}")
        
        return result
    
    def evaluate_numeric(self, feature_name: str) -> Dict:
        """Đánh giá đặc trưng số (ví dụ: size)"""
        print(f"\n📏 Đánh giá {feature_name}...")
        
        y_true = []
        y_pred = []
        
        for item in self.matched_data:
            # Lấy size từ ground truth
            if feature_name == 'size':
                gt = item['ground_truth'].get('size_um', 
                                             item['ground_truth'].get('size', None))
                
                # Lấy size từ prediction
                pred_dict = item['prediction'].get('size', {})
                if isinstance(pred_dict, dict):
                    pred = pred_dict.get('metrics', {}).get('diameter_micron', None)
                else:
                    pred = None
                
                if gt and pred:
                    y_true.append(float(gt))
                    y_pred.append(float(pred))
        
        if not y_true:
            return {'error': 'No data to evaluate'}
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Tính metrics
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        bias = np.mean(y_pred - y_true)
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        result = {
            'mae': mae,
            'rmse': rmse,
            'bias': bias,
            'r2': r2,
            'mean_true': np.mean(y_true),
            'mean_pred': np.mean(y_pred),
            'num_samples': len(y_true)
        }
        
        # In kết quả
        print(f"  MAE: {mae:.2f} μm")
        print(f"  RMSE: {rmse:.2f} μm")
        print(f"  R²: {r2:.3f}")
        print(f"  Bias: {bias:.2f} μm")
        
        return result
    
    def _accuracy(self, y_true: List, y_pred: List) -> float:
        """Tính accuracy"""
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true) if y_true else 0
    
    def _precision_recall_f1(self, y_true: List, y_pred: List) -> Tuple[Dict, Dict, Dict]:
        """Tính precision, recall, F1 cho mỗi class"""
        classes = set(y_true + y_pred)
        
        precision = {}
        recall = {}
        f1 = {}
        
        for cls in classes:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
            
            precision[cls] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            if precision[cls] + recall[cls] > 0:
                f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls])
            else:
                f1[cls] = 0
        
        return precision, recall, f1
    
    def _confusion_matrix(self, y_true: List, y_pred: List) -> Dict:
        """Tạo confusion matrix"""
        classes = sorted(set(y_true + y_pred))
        matrix = defaultdict(lambda: defaultdict(int))
        
        for t, p in zip(y_true, y_pred):
            matrix[t][p] += 1
        
        return {
            'classes': classes,
            'matrix': {t: {p: matrix[t][p] for p in classes} for t in classes}
        }
    
    def _per_class_metrics(self, y_true: List, y_pred: List) -> Dict:
        """Tính metrics chi tiết cho từng class"""
        classes = sorted(set(y_true + y_pred))
        
        true_counts = Counter(y_true)
        pred_counts = Counter(y_pred)
        
        metrics = {}
        for cls in classes:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
            
            metrics[cls] = {
                'support': true_counts[cls],
                'predicted': pred_counts[cls],
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        return metrics
    
    def compute_summary(self) -> Dict:
        """Tính tổng quan kết quả"""
        summary = {
            'total_samples': len(self.matched_data),
            'features_evaluated': []
        }
        
        # Tính trung bình các metrics
        avg_accuracy = []
        avg_f1 = []
        
        for feature in self.categorical_features:
            if feature in self.results and 'accuracy' in self.results[feature]:
                avg_accuracy.append(self.results[feature]['accuracy'])
                avg_f1.append(self.results[feature]['macro_f1'])
                summary['features_evaluated'].append(feature)
        
        if avg_accuracy:
            summary['overall_accuracy'] = np.mean(avg_accuracy)
            summary['overall_macro_f1'] = np.mean(avg_f1)
        
        return summary
    
    def generate_report(self, output_file: str = None):
        """Tạo báo cáo chi tiết"""
        print("\n" + "="*60)
        print("BÁO CÁO ĐÁNH GIÁ ĐỊNH LƯỢNG")
        print("="*60)
        
        if 'summary' in self.results:
            summary = self.results['summary']
            print(f"\n📋 TỔNG QUAN:")
            print(f"  Tổng số mẫu đánh giá: {summary['total_samples']}")
            if 'overall_accuracy' in summary:
                print(f"  Accuracy trung bình: {summary['overall_accuracy']:.2%}")
                print(f"  Macro F1 trung bình: {summary['overall_macro_f1']:.2%}")
        
        # Chi tiết từng đặc trưng
        for feature in self.categorical_features:
            if feature in self.results and 'accuracy' in self.results[feature]:
                result = self.results[feature]
                print(f"\n📊 {feature.upper()}:")
                print(f"  Accuracy: {result['accuracy']:.2%}")
                print(f"  Macro Precision: {result['macro_precision']:.2%}")
                print(f"  Macro Recall: {result['macro_recall']:.2%}")
                print(f"  Macro F1: {result['macro_f1']:.2%}")
                
                # Top 3 best and worst classes
                f1_scores = result['f1']
                if f1_scores:
                    sorted_f1 = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
                    print(f"  Best class: {sorted_f1[0][0]} (F1={sorted_f1[0][1]:.2%})")
                    if len(sorted_f1) > 1:
                        print(f"  Worst class: {sorted_f1[-1][0]} (F1={sorted_f1[-1][1]:.2%})")
        
        for feature in self.numeric_features:
            if feature in self.results and 'mae' in self.results[feature]:
                result = self.results[feature]
                print(f"\n📏 {feature.upper()}:")
                print(f"  MAE: {result['mae']:.2f} μm")
                print(f"  RMSE: {result['rmse']:.2f} μm")
                print(f"  R²: {result['r2']:.3f}")
                print(f"  Bias: {result['bias']:.2f} μm")
        
        # Lưu vào file nếu cần
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"\n✅ Đã lưu báo cáo chi tiết tại: {output_file}")
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Xuất kết quả chi tiết ra DataFrame để phân tích"""
        rows = []
        
        for item in self.matched_data:
            row = {
                'image': os.path.basename(item['prediction'].get('image_path', '')),
            }
            
            # Ground truth
            for feature in self.categorical_features:
                row[f'gt_{feature}'] = item['ground_truth'].get(feature, None)
            
            # Predictions
            for feature in self.categorical_features:
                pred_dict = item['prediction'].get(feature, {})
                if isinstance(pred_dict, dict):
                    row[f'pred_{feature}'] = pred_dict.get(f'{feature}_class', 
                                                           pred_dict.get('class', None))
                    row[f'conf_{feature}'] = pred_dict.get('confidence', None)
                else:
                    row[f'pred_{feature}'] = pred_dict
            
            # Size
            row['gt_size'] = item['ground_truth'].get('size_um', 
                                                      item['ground_truth'].get('size', None))
            size_dict = item['prediction'].get('size', {})
            if isinstance(size_dict, dict):
                row['pred_size'] = size_dict.get('metrics', {}).get('diameter_micron', None)
            
            rows.append(row)
        
        return pd.DataFrame(rows)


def main():
    """Hàm main để chạy evaluation standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Đánh giá kết quả trích xuất đặc trưng')
    parser.add_argument('--pred', required=True, help='File JSON chứa predictions')
    parser.add_argument('--gt', required=True, help='File JSON chứa ground truth')
    parser.add_argument('--output', '-o', help='File lưu báo cáo (JSON)')
    parser.add_argument('--csv', help='File lưu chi tiết (CSV)')
    
    args = parser.parse_args()
    
    # Chạy evaluation
    evaluator = PollenEvaluator(args.pred, args.gt)
    evaluator.evaluate_all()
    evaluator.generate_report(args.output)
    
    # Xuất CSV nếu cần
    if args.csv:
        df = evaluator.export_to_dataframe()
        df.to_csv(args.csv, index=False, encoding='utf-8-sig')
        print(f"✅ Đã lưu chi tiết tại: {args.csv}")


if __name__ == '__main__':
    main()
