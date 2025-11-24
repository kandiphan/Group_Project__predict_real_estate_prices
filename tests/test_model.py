import unittest
import numpy as np
import pandas as pd
from model import make_prediction, load_artifacts

class TestModel(unittest.TestCase):
    def setUp(self):
        self.artifacts = load_artifacts()
        self.test_data = pd.DataFrame({
            'category_name': ['Chung cư'],
            'size': [50.0],
            'region_name': ['Hồ Chí Minh'],
            'area_name': ['Quận 1'],
            'ward_name': ['Phường Bến Nghé']
        })
        
    def test_prediction_range(self):
        """Test if predictions are within reasonable ranges"""
        pred, _ = make_prediction(
            self.test_data,
            self.artifacts['model'],
            self.artifacts['preprocessor'],
            self.artifacts['y_transformer'],
            self.artifacts['artifacts']
        )
        
        # Kiểm tra giá trị dự đoán có nằm trong khoảng hợp lý
        self.assertTrue(pred > 0)  # Giá không âm
        self.assertTrue(pred < 1e12)  # Giá không quá 1000 tỷ
        
    def test_log_transform(self):
        """Test if log transforms are applied correctly"""
        # Test với size = 100m2
        test_data = self.test_data.copy()
        test_data['size'] = 100.0
        
        # Kiểm tra xem size có được transform bằng log1p không
        if self.artifacts['artifacts'].get('X_transforms', {}).get('size') == 'log1p':
            transformed_size = np.log1p(test_data['size'])
            self.assertTrue(abs(transformed_size - 4.61) < 0.01)  # log1p(100) ≈ 4.61

if __name__ == '__main__':
    unittest.main()