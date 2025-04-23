import unittest
import numpy as np
import cv2

from src.ethnicity_detection.cnn_classifier import CNNEthnicityClassifier
from src.config import ETHNICITY_MAPPING_REVERSE

class TestEthnicityDetection(unittest.TestCase):
    def setUp(self):
        # Buat gambar uji
        self.test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.rectangle(self.test_image, (50, 50), (150, 150), (255, 255, 255), -1)

    def test_ethnicity_prediction(self):
        classifier = CNNEthnicityClassifier()
        
        # Uji prediksi etnis
        class_idx, probabilities = classifier.predict(self.test_image)
        
        # Periksa keluaran
        self.assertIsNotNone(class_idx)
        self.assertEqual(len(probabilities), len(ETHNICITY_MAPPING_REVERSE))
        
        # Periksa probabilitas valid
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))
        self.assertAlmostEqual(np.sum(probabilities), 1.0, places=6)
    
    def test_top_n_predictions(self):
        classifier = CNNEthnicityClassifier()
        
        # Uji prediksi top-N
        top_predictions = classifier.predict_top_n(self.test_image, n=3)
        
        # Periksa jumlah prediksi
        self.assertEqual(len(top_predictions), 3)
        
        # Periksa format prediksi
        for idx, prob in top_predictions:
            self.assertIn(idx, range(len(ETHNICITY_MAPPING_REVERSE)))
            self.assertTrue(0 <= prob <= 1)

if __name__ == '__main__':
    unittest.main()