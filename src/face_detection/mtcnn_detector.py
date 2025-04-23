import numpy as np
from mtcnn import MTCNN
from src.face_detection.detector import FaceDetector
from src.config import FACE_DETECTION_CONFIDENCE_THRESHOLD

class MTCNNDetector(FaceDetector):
    """Face detector using MTCNN"""
    
    def __init__(self, min_confidence=FACE_DETECTION_CONFIDENCE_THRESHOLD):
        """
        Initialize MTCNN detector
        
        Args:
            min_confidence: Minimum confidence threshold for detections
        """
        # Update inisialisasi MTCNN
        self.detector = MTCNN()  # Hapus parameter yang tidak didukung
        self.min_confidence = min_confidence
    
    def detect_faces(self, image):
        """
        Detect faces in an image using MTCNN
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            list: List of dictionaries, each containing:
                - 'box': [x, y, width, height]
                - 'confidence': Detection confidence
                - 'landmarks': Face landmarks
        """
        detections = self.detector.detect_faces(image)
        
        # Filter by confidence
        detections = [d for d in detections if d['confidence'] >= self.min_confidence]
        
        return detections