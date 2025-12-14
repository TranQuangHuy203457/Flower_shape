import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, IMAGE_CONFIG, FEATURE_TYPES, MODELS_DIR
from pollen_features.database_handler import DatabaseHandler


class PollenDataset(Dataset):
    """Dataset cho ảnh phấn hoa"""
    
    def __init__(self, image_paths: List[str], labels: List[str], 
                 transform=None, label_encoder=None):
        """
        Args:
            image_paths: Danh sách đường dẫn ảnh
            labels: Danh sách nhãn tương ứng
            transform: Các phép biến đổi ảnh
            label_encoder: Bộ mã hóa nhãn
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.encoded_labels = self.label_encoder.fit_transform(labels)
        else:
            self.label_encoder = label_encoder
            self.encoded_labels = self.label_encoder.transform(labels)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Đọc ảnh
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.encoded_labels[idx]
        
        return image, label


class PollenClassifier(nn.Module):
    """Model phân loại đặc trưng phấn hoa"""
    
    def __init__(self, num_classes: int, backbone: str = "resnet50", 
                 pretrained: bool = True, dropout_rate: float = 0.5):
        super(PollenClassifier, self).__init__()
        
        # Chọn backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "vgg16":
            self.backbone = models.vgg16(pretrained=pretrained)
            num_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Backbone không được hỗ trợ: {backbone}")
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output
    
    def extract_features(self, x):
        """Trích xuất features từ backbone"""
        with torch.no_grad():
            features = self.backbone(x)
            if len(features.shape) > 2:
                features = features.view(features.size(0), -1)
        return features


class PollenTrainer:
    """Trainer cho model phân loại phấn hoa"""
    
    def __init__(self, feature_type: str, db_handler: DatabaseHandler = None):
        """
        Args:
            feature_type: Loại đặc trưng cần train (shape, size, surface, etc.)
            db_handler: Database handler để lấy dữ liệu
        """
        self.feature_type = feature_type
        self.db_handler = db_handler or DatabaseHandler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.label_encoder = None
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        
        # Chuẩn bị transforms
        self.train_transform = transforms.Compose([
            transforms.Resize(IMAGE_CONFIG["target_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(IMAGE_CONFIG["target_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def prepare_data(self, test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        # Lấy dữ liệu từ database
        images = self.db_handler.get_all_images()
        
        image_paths = []
        labels = []
        
        for img in images:
            # Lấy nhãn từ ground truth
            conn = self.db_handler._get_connection()
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT {self.feature_type}_label 
                FROM ground_truth_labels 
                WHERE image_id = ? AND {self.feature_type}_label IS NOT NULL
            ''', (img['id'],))
            row = cursor.fetchone()
            
            if row and os.path.exists(img['image_path']):
                image_paths.append(img['image_path'])
                labels.append(row[0])
        
        if len(image_paths) == 0:
            raise ValueError(f"Không có dữ liệu training cho {self.feature_type}")
        
        print(f"Tổng số mẫu: {len(image_paths)}")
        print(f"Phân phối nhãn: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        # Chia train/test
        X_train, X_test, y_train, y_test = train_test_split(
            image_paths, labels, test_size=test_size, stratify=labels, random_state=42
        )
        
        # Chia train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size/(1-test_size), 
            stratify=y_train, random_state=42
        )
        
        # Tạo label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        
        # Tạo datasets
        train_dataset = PollenDataset(X_train, y_train, self.train_transform, self.label_encoder)
        val_dataset = PollenDataset(X_val, y_val, self.val_transform, self.label_encoder)
        test_dataset = PollenDataset(X_test, y_test, self.val_transform, self.label_encoder)
        
        # Tạo dataloaders
        batch_size = MODEL_CONFIG["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return train_loader, val_loader, test_loader
    
    def prepare_data_from_folder(self, data_dir: str, test_size: float = 0.2, 
                                  val_size: float = 0.1) -> Tuple:
        image_paths = []
        labels = []
        
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_paths.append(os.path.join(class_dir, img_name))
                        labels.append(class_name)
        
        print(f"Tổng số mẫu: {len(image_paths)}")
        print(f"Số lớp: {len(set(labels))}")
        
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            image_paths, labels, test_size=test_size, stratify=labels, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size/(1-test_size), 
            stratify=y_train, random_state=42
        )
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        
        train_dataset = PollenDataset(X_train, y_train, self.train_transform, self.label_encoder)
        val_dataset = PollenDataset(X_val, y_val, self.val_transform, self.label_encoder)
        test_dataset = PollenDataset(X_test, y_test, self.val_transform, self.label_encoder)
        
        batch_size = MODEL_CONFIG["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return train_loader, val_loader, test_loader
    
    def build_model(self, num_classes: int = None):
        """Xây dựng model"""
        if num_classes is None:
            num_classes = len(self.label_encoder.classes_)
        
        self.model = PollenClassifier(
            num_classes=num_classes,
            backbone=MODEL_CONFIG["backbone"],
            pretrained=MODEL_CONFIG["pretrained"],
            dropout_rate=MODEL_CONFIG["dropout_rate"]
        )
        self.model = self.model.to(self.device)
        
        print(f"Model được tạo với {num_classes} lớp trên {self.device}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = None, learning_rate: float = None):
        num_epochs = num_epochs or MODEL_CONFIG["num_epochs"]
        learning_rate = learning_rate or MODEL_CONFIG["learning_rate"]
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          factor=0.5, patience=5)
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': train_loss / (pbar.n + 1),
                    'acc': 100. * train_correct / train_total
                })
            
            train_acc = 100. * train_correct / train_total
            train_loss = train_loss / len(train_loader)
            
            # Validation phase
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Lưu history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"\nEpoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Lưu model tốt nhất
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                print(f"  -> Best model saved! (Val Acc: {best_val_acc:.2f}%)")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return self.history
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Đánh giá model trên một dataset"""
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(data_loader)
        
        return avg_loss, accuracy
    
    def test(self, test_loader: DataLoader) -> Dict:
        """Đánh giá chi tiết trên test set"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Classification report
        class_names = self.label_encoder.classes_
        report = classification_report(all_labels, all_preds, 
                                        target_names=class_names, 
                                        output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            "classification_report": report,
            "confusion_matrix": cm,
            "class_names": list(class_names),
            "predictions": all_preds,
            "labels": all_labels
        }
    
    def save_model(self, save_path: str = None) -> str:
        """Lưu model"""
        if save_path is None:
            save_path = os.path.join(MODELS_DIR, f"{self.feature_type}_classifier.pth")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'label_encoder_classes': list(self.label_encoder.classes_),
            'feature_type': self.feature_type,
            'history': self.history,
            'config': MODEL_CONFIG
        }
        
        torch.save(checkpoint, save_path)
        print(f"Model đã được lưu tại: {save_path}")
        
        return save_path
    
    def load_model(self, model_path: str):
        """Load model đã train"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Khôi phục label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(checkpoint['label_encoder_classes'])
        
        # Xây dựng và load model
        num_classes = len(self.label_encoder.classes_)
        self.build_model(num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.feature_type = checkpoint.get('feature_type', self.feature_type)
        self.history = checkpoint.get('history', self.history)
        
        print(f"Model loaded từ: {model_path}")
    
    def predict(self, image_path: str) -> Dict:
        """Dự đoán cho một ảnh"""
        self.model.eval()
        
        # Load và transform ảnh
        image = Image.open(image_path).convert('RGB')
        image = self.val_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        class_name = self.label_encoder.inverse_transform([predicted.item()])[0]
        
        # Lấy top-k predictions
        top_k = min(3, len(self.label_encoder.classes_))
        top_probs, top_indices = probabilities.topk(top_k)
        
        top_predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            top_predictions.append({
                'class': self.label_encoder.inverse_transform([idx.item()])[0],
                'probability': prob.item()
            })
        
        return {
            'predicted_class': class_name,
            'confidence': confidence.item(),
            'top_predictions': top_predictions
        }
    
    def plot_training_history(self, save_path: str = None):
        """Vẽ biểu đồ training history"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Accuracy')
        axes[1].plot(self.history['val_acc'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Biểu đồ đã lưu tại: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, test_results: Dict, save_path: str = None):
        """Vẽ confusion matrix"""
        cm = test_results['confusion_matrix']
        class_names = test_results['class_names']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {self.feature_type.upper()}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


def train_all_features(data_dir: str):
    """Train models cho tất cả các loại đặc trưng"""
    results = {}
    
    for feature_type in FEATURE_TYPES.keys():
        feature_data_dir = os.path.join(data_dir, feature_type)
        
        if not os.path.exists(feature_data_dir):
            print(f"Bỏ qua {feature_type}: không tìm thấy dữ liệu")
            continue
        
        print(f"\n{'='*50}")
        print(f"Training model cho: {feature_type.upper()}")
        print(f"{'='*50}")
        
        try:
            trainer = PollenTrainer(feature_type)
            train_loader, val_loader, test_loader = trainer.prepare_data_from_folder(feature_data_dir)
            trainer.build_model()
            trainer.train(train_loader, val_loader)
            
            # Đánh giá
            test_results = trainer.test(test_loader)
            
            # Lưu model
            model_path = trainer.save_model()
            
            results[feature_type] = {
                'model_path': model_path,
                'test_accuracy': test_results['classification_report']['accuracy'],
                'report': test_results['classification_report']
            }
            
            # Vẽ biểu đồ
            trainer.plot_training_history(
                os.path.join(MODELS_DIR, f"{feature_type}_history.png")
            )
            
        except Exception as e:
            print(f"Lỗi khi train {feature_type}: {e}")
            results[feature_type] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Pollen Feature Classifier')
    parser.add_argument('--feature', type=str, default='shape',
                        choices=list(FEATURE_TYPES.keys()),
                        help='Loại đặc trưng cần train')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Thư mục chứa dữ liệu training')
    parser.add_argument('--epochs', type=int, default=MODEL_CONFIG['num_epochs'],
                        help='Số epoch')
    parser.add_argument('--batch_size', type=int, default=MODEL_CONFIG['batch_size'],
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=MODEL_CONFIG['learning_rate'],
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Cập nhật config
    MODEL_CONFIG['num_epochs'] = args.epochs
    MODEL_CONFIG['batch_size'] = args.batch_size
    MODEL_CONFIG['learning_rate'] = args.lr
    
    # Train
    trainer = PollenTrainer(args.feature)
    train_loader, val_loader, test_loader = trainer.prepare_data_from_folder(args.data_dir)
    trainer.build_model()
    trainer.train(train_loader, val_loader)
    
    # Test
    test_results = trainer.test(test_loader)
    print("\nClassification Report:")
    print(json.dumps(test_results['classification_report'], indent=2))
    
    # Lưu
    trainer.save_model()
    trainer.plot_training_history()
    trainer.plot_confusion_matrix(test_results)
