from abc import ABC, abstractmethod
import numpy as np

class FaceDetector(ABC):
    """Abstract base class for face detectors"""
    
    @abstractmethod
    def detect_faces(self, image):
        """
        Detect faces in an image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            list: List of dictionaries, each containing:
                - 'box': [x, y, width, height]
                - 'confidence': Detection confidence
                - 'landmarks': Face landmarks (if available)
        """
        pass
    
    def extract_faces(self, image, padding=0.2):
        """
        Detect and extract face regions from image
        
        Args:
            image: Input image
            padding: Padding around the face (percentage of face size)
            
        Returns:
            list: List of dictionaries, each containing:
                - 'face': Cropped face image
                - 'box': [x, y, width, height]
                - 'confidence': Detection confidence
                - 'landmarks': Face landmarks (if available)
        """
        detections = self.detect_faces(image)
        faces = []
        
        for detection in detections:
            box = detection['box']
            x, y, w, h = box
            
            # Add padding
            pad_w = int(w * padding)
            pad_h = int(h * padding)
            
            # Calculate new box with padding
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(image.shape[1], x + w + pad_w)
            y2 = min(image.shape[0], y + h + pad_h)
            
            # Extract face
            face = image[y1:y2, x1:x2]
            faces.append({
                'face': face,
                'box': [x1, y1, x2-x1, y2-y1],
                'confidence': detection['confidence'],
                'landmarks': detection.get('landmarks', None)
            })
            
        return faces