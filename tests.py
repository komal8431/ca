import unittest
import numpy as np
import sys
sys.path.append('.')  
from model.model import generate_product_data, preprocess_data, apply_automl  # Assuming your script is named 'product_price_prediction_script.py'

class TestProductPricePrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Generate and preprocess the data once for all tests"""
        cls.product_data = generate_product_data()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = preprocess_data(cls.product_data)
        cls.best_models, cls.best_r2_scores, cls.predictions = apply_automl(cls.X_train, cls.X_test, cls.y_train, cls.y_test)

    def test_model_accuracy(self):
        """Test that at least one model has a good R-squared score"""
        # Define a reasonable threshold for R-squared score
        r2_threshold = 0.6
        best_r2 = max(self.best_r2_scores.values())
        self.assertTrue(best_r2 > r2_threshold, f"Model accuracy is below threshold: {best_r2}")

    def test_no_missing_values(self):
        """Test that the dataset has no missing values"""
        missing_values = self.product_data.isnull().sum().sum()
        self.assertEqual(missing_values, 0, "Dataset contains missing values")

    def test_correct_price_predictions(self):
        """Test that predicted prices are within a reasonable range based on historical data"""
        y_test_min, y_test_max = self.y_test.min(), self.y_test.max()
        for model_name, y_pred in self.predictions.items():
            within_range = np.all((y_pred >= y_test_min) & (y_pred <= y_test_max))
            self.assertTrue(within_range, f"Predictions for {model_name} fall outside of expected price range.")

# Run the unit tests
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)


