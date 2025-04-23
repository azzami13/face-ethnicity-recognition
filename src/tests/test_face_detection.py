import unittest
import numpy as np
import cv2

from src.face_detection.mtcnn_detector import MTCNNDetector
from src.face_detection.retinaface_detector import RetinaFaceDetector
from src.config import FACE_DETECTION_CONFIDENCE_THRESHOLD

class TestFaceDetection(unittest.TestCase):
    def setUp(self):
        """
        Persiapan untuk setiap test case
        """
        # Buat gambar uji sederhana
        self.test_image = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Tambahkan wajah palsu
        cv2.rectangle(self.test_image, (100, 100), (200, 200), (255, 255, 255), -1)
    
    def test_mtcnn_detector_initialization(self):
        """
        Uji inisialisasi MTCNN detector
        """
        detector = MTCNNDetector()
        self.assertIsNotNone(detector)
        self.assertTrue(hasattr(detector, 'detect_faces'))
    
    def test_mtcnn_face_detection(self):
        """
        Uji deteksi wajah dengan MTCNN
        """
        detector = MTCNNDetector()
        faces = detector.detect_faces(self.test_image)
        
        # Pastikan deteksi berhasil
        self.assertIsNotNone(faces)
        
        # Pastikan deteksi memiliki confidence yang memadai
        for face in faces:
            self.assertIn('confidence', face)
            self.assertGreaterEqual(face['confidence'], FACE_DETECTION_CONFIDENCE_THRESHOLD)
    
    def test_mtcnn_face_extraction(self):
        """
        Uji ekstraksi wajah dengan MTCNN
        """
        detector = MTCNNDetector()
        extracted_faces = detector.extract_faces(self.test_image)
        
        # Pastikan ekstraksi berhasil
        self.assertIsNotNone(extracted_faces)
        
        # Periksa struktur setiap wajah yang diekstrak
        for face_data in extracted_faces:
            self.assertIn('face', face_data)
            self.assertIn('box', face_data)
            self.assertIn('confidence', face_data)
            
            # Pastikan ukuran wajah valid
            self.assertTrue(face_data['face'].size > 0)
    
    def test_retinaface_detector(self):
        """
        Uji deteksi wajah dengan RetinaFace
        """
        detector = RetinaFaceDetector()
        faces = detector.detect_faces(self.test_image)
        
        # Pastikan deteksi berhasil
        self.assertIsNotNone(faces)
        
        # Pastikan deteksi memiliki confidence yang memadai
        for face in faces:
            self.assertIn('confidence', face)
    
    def test_multiple_detector_comparison(self):
        """
        Perbandingan antara MTCNN dan RetinaFace
        """
        mtcnn_detector = MTCNNDetector()
        retinaface_detector = RetinaFaceDetector()
        
        mtcnn_faces = mtcnn_detector.detect_faces(self.test_image)
        retinaface_faces = retinaface_detector.detect_faces(self.test_image)
        
        # Bandingkan jumlah deteksi
        print(f"MTCNN Faces: {len(mtcnn_faces)}")
        print(f"RetinaFace Faces: {len(retinaface_faces)}")

if __name__ == '__main__':
    unittest.main()