import sqlite3
import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE_CONFIG, FEATURE_TYPES


class DatabaseHandler:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DATABASE_CONFIG["path"]
        self.connection = None
        self._create_tables()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Lấy kết nối database"""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
        return self.connection
    
    def _create_tables(self):
        """Tạo các bảng cần thiết trong database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Bảng lưu thông tin ảnh phấn hoa
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pollen_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT UNIQUE NOT NULL,
                image_name TEXT,
                species_name TEXT,
                family_name TEXT,
                genus_name TEXT,
                source TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Bảng lưu đặc trưng đã trích xuất
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pollen_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                shape_class TEXT,
                shape_confidence REAL,
                shape_metrics TEXT,
                size_class TEXT,
                size_metrics TEXT,
                surface_class TEXT,
                surface_confidence REAL,
                surface_metrics TEXT,
                aperture_class TEXT,
                aperture_metrics TEXT,
                exine_class TEXT,
                exine_metrics TEXT,
                section_class TEXT,
                section_metrics TEXT,
                deep_features BLOB,
                extraction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (image_id) REFERENCES pollen_images (id)
            )
        ''')
        
        # Bảng lưu nhãn ground truth (để train)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ground_truth_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                shape_label TEXT,
                size_label TEXT,
                surface_label TEXT,
                aperture_label TEXT,
                exine_label TEXT,
                section_label TEXT,
                labeled_by TEXT,
                labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                verified BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (image_id) REFERENCES pollen_images (id)
            )
        ''')
        
        # Bảng lưu thông tin training
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                model_path TEXT,
                feature_type TEXT,
                train_accuracy REAL,
                val_accuracy REAL,
                test_accuracy REAL,
                num_epochs INTEGER,
                batch_size INTEGER,
                learning_rate REAL,
                config TEXT,
                trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
    
    def add_image(self, image_path: str, species_name: str = None, 
                  family_name: str = None, genus_name: str = None,
                  source: str = None, notes: str = None) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        image_name = os.path.basename(image_path)
        
        try:
            cursor.execute('''
                INSERT INTO pollen_images 
                (image_path, image_name, species_name, family_name, genus_name, source, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (image_path, image_name, species_name, family_name, genus_name, source, notes))
            
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Ảnh đã tồn tại, lấy ID
            cursor.execute('SELECT id FROM pollen_images WHERE image_path = ?', (image_path,))
            return cursor.fetchone()[0]
    
    def add_features(self, image_id: int, features: Dict) -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Serialize các metrics thành JSON
        shape = features.get("shape", {})
        size = features.get("size", {})
        surface = features.get("surface", {})
        aperture = features.get("aperture_type", {})
        exine = features.get("exine", {})
        section = features.get("section", {})
        
        # Serialize deep features thành bytes
        deep_features = features.get("deep_features")
        if deep_features is not None:
            deep_features_blob = deep_features.tobytes()
        else:
            deep_features_blob = None
        
        cursor.execute('''
            INSERT INTO pollen_features 
            (image_id, shape_class, shape_confidence, shape_metrics,
             size_class, size_metrics, surface_class, surface_confidence,
             surface_metrics, aperture_class, aperture_metrics,
             exine_class, exine_metrics, section_class, section_metrics,
             deep_features)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            image_id,
            shape.get("shape_class"),
            shape.get("confidence"),
            json.dumps(shape.get("metrics", {})),
            size.get("size_class"),
            json.dumps(size.get("metrics", {})),
            surface.get("surface_class"),
            surface.get("confidence"),
            json.dumps(surface.get("metrics", {})),
            aperture.get("aperture_class"),
            json.dumps(aperture.get("metrics", {})),
            exine.get("exine_class"),
            json.dumps(exine.get("metrics", {})),
            section.get("section_class"),
            json.dumps(section.get("metrics", {})),
            deep_features_blob
        ))
        
        conn.commit()
        return cursor.lastrowid
    
    def add_ground_truth(self, image_id: int, labels: Dict, labeled_by: str = "manual") -> int:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ground_truth_labels 
            (image_id, shape_label, size_label, surface_label, 
             aperture_label, exine_label, section_label, labeled_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            image_id,
            labels.get("shape"),
            labels.get("size"),
            labels.get("surface"),
            labels.get("aperture_type"),
            labels.get("exine"),
            labels.get("section"),
            labeled_by
        ))
        
        conn.commit()
        return cursor.lastrowid
    
    def get_image(self, image_id: int) -> Optional[Dict]:
        """Lấy thông tin ảnh theo ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM pollen_images WHERE id = ?', (image_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_features(self, image_id: int) -> Optional[Dict]:
        """Lấy đặc trưng của ảnh theo ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM pollen_features WHERE image_id = ?', (image_id,))
        row = cursor.fetchone()
        
        if row:
            result = dict(row)
            # Parse JSON metrics
            for key in ['shape_metrics', 'size_metrics', 'surface_metrics', 
                       'aperture_metrics', 'exine_metrics', 'section_metrics']:
                if result[key]:
                    result[key] = json.loads(result[key])
            return result
        return None
    
    def get_all_images(self, species: str = None, family: str = None) -> List[Dict]:
        """Lấy tất cả ảnh, có thể lọc theo loài hoặc họ"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        query = 'SELECT * FROM pollen_images WHERE 1=1'
        params = []
        
        if species:
            query += ' AND species_name = ?'
            params.append(species)
        if family:
            query += ' AND family_name = ?'
            params.append(family)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_training_data(self, feature_type: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Join features và ground truth
        cursor.execute('''
            SELECT f.deep_features, g.{}_label, f.image_id
            FROM pollen_features f
            JOIN ground_truth_labels g ON f.image_id = g.image_id
            WHERE g.{}_label IS NOT NULL AND f.deep_features IS NOT NULL
        '''.format(feature_type, feature_type))
        
        rows = cursor.fetchall()
        
        if not rows:
            return np.array([]), np.array([]), []
        
        features = []
        labels = []
        image_ids = []
        
        for row in rows:
            # Deserialize deep features
            deep_features = np.frombuffer(row[0], dtype=np.float32)
            features.append(deep_features)
            labels.append(row[1])
            image_ids.append(row[2])
        
        return np.array(features), np.array(labels), image_ids
    
    def export_to_csv(self, output_path: str = None) -> str:
        """Xuất dữ liệu ra file CSV"""
        output_path = output_path or DATABASE_CONFIG["csv_export"]
        
        conn = self._get_connection()
        
        # Query tất cả dữ liệu
        query = '''
            SELECT 
                i.id, i.image_path, i.image_name, i.species_name, i.family_name,
                f.shape_class, f.shape_confidence,
                f.size_class, f.surface_class, f.surface_confidence,
                f.aperture_class, f.exine_class, f.section_class,
                g.shape_label, g.size_label, g.surface_label,
                g.aperture_label, g.exine_label, g.section_label
            FROM pollen_images i
            LEFT JOIN pollen_features f ON i.id = f.image_id
            LEFT JOIN ground_truth_labels g ON i.id = g.image_id
        '''
        
        df = pd.read_sql_query(query, conn)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        return output_path
    
    def import_from_csv(self, csv_path: str, has_labels: bool = True):
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            # Thêm ảnh
            image_id = self.add_image(
                image_path=row.get('image_path', ''),
                species_name=row.get('species_name'),
                family_name=row.get('family_name'),
                genus_name=row.get('genus_name'),
                source=row.get('source'),
                notes=row.get('notes')
            )
            
            # Thêm labels nếu có
            if has_labels:
                labels = {
                    'shape': row.get('shape_label'),
                    'size': row.get('size_label'),
                    'surface': row.get('surface_label'),
                    'aperture_type': row.get('aperture_label'),
                    'exine': row.get('exine_label'),
                    'section': row.get('section_label'),
                }
                # Chỉ thêm nếu có ít nhất 1 label
                if any(v for v in labels.values() if pd.notna(v)):
                    self.add_ground_truth(image_id, labels)
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê về database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Tổng số ảnh
        cursor.execute('SELECT COUNT(*) FROM pollen_images')
        stats['total_images'] = cursor.fetchone()[0]
        
        # Số ảnh đã có đặc trưng
        cursor.execute('SELECT COUNT(DISTINCT image_id) FROM pollen_features')
        stats['images_with_features'] = cursor.fetchone()[0]
        
        # Số ảnh đã có nhãn
        cursor.execute('SELECT COUNT(DISTINCT image_id) FROM ground_truth_labels')
        stats['images_with_labels'] = cursor.fetchone()[0]
        
        # Thống kê theo loài
        cursor.execute('''
            SELECT species_name, COUNT(*) as count 
            FROM pollen_images 
            WHERE species_name IS NOT NULL
            GROUP BY species_name
        ''')
        stats['by_species'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Phân phối các lớp đặc trưng
        for feature_type in FEATURE_TYPES.keys():
            cursor.execute(f'''
                SELECT {feature_type}_class, COUNT(*) 
                FROM pollen_features 
                WHERE {feature_type}_class IS NOT NULL
                GROUP BY {feature_type}_class
            ''')
            stats[f'{feature_type}_distribution'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        return stats
    
    def close(self):
        """Đóng kết nối database"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    # Test database handler
    with DatabaseHandler() as db:
        # Thêm ảnh test
        image_id = db.add_image(
            image_path="/path/to/test/image.jpg",
            species_name="Helianthus annuus",
            family_name="Asteraceae"
        )
        print(f"Added image with ID: {image_id}")
        
        # Thêm ground truth
        labels = {
            "shape": "spherical",
            "size": "medium",
            "surface": "echinate",
            "aperture_type": "tricolporate",
            "exine": "thick",
            "section": "equatorial"
        }
        label_id = db.add_ground_truth(image_id, labels)
        print(f"Added labels with ID: {label_id}")
        
        # Lấy thống kê
        stats = db.get_statistics()
        print(f"Database statistics: {stats}")
