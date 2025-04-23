import unittest
import numpy as np
import cv2

from src.face_similarity.facenet_embedder import FaceNetEmbedder

class TestFaceSimilarity(unittest.TestCase):
    def setUp(self):
        # Buat gambar uji
        self.test_image1 = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.rectangle(self.test_image1, (50, 50), (150, 150), (255, 255, 255), -1)
        
        self.test_image2 = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.rectangle(self.test_image2, (50, 50), (150, 150), (255, 255, 255), -1)
        
        self.test_image3 = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.rectangle(self.test_image3, (10, 10), (100, 100), (255, 255, 255), -1)

    def test_embedding_generation(self):
        embedder = FaceNetEmbedder()
        
        # Uji pembuatan embedding
        embedding1 = embedder.get_embedding(self.test_image1)
        embedding2 = embedder.get_embedding(self.test_image2)
        embedding3 = embedder.get_embedding(self.test_image3)
        
        # Periksa ukuran embedding
        self.assertEqual(embedding1.shape[0], 512)
        self.assertEqual(embedding2.shape[0], 512)
        self.assertEqual(embedding3.shape[0], 512)
    
    def test_face_comparison(self):
        embedder = FaceNetEmbedder()
        
        # Bandingkan gambar yang sama
        similarity_same = embedder.compare_faces(self.test_image1, self.test_image2)
        
        # Bandingkan gambar berbeda
        similarity_different = embedder.compare_faces(self.test_image1, self.test_image3)
        
        # Periksa skor kemiripan
        self.assertGreater(similarity_same, similarity_different)
        self.assertGreaterEqual(similarity_same, 0)
        self.assertLessEqual(similarity_same, 1)

if __name__ == '__main__':
    unittest.main()