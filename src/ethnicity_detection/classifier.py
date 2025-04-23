from abc import ABC, abstractmethod
import numpy as np

class EthnicityClassifier(ABC):
    """Abstract base class for ethnicity classifiers"""
    
    @abstractmethod
    def predict(self, face_image):
        """
        Predict ethnicity for a face image
        
        Args:
            face_image: Face image (already cropped and aligned)
            
        Returns:
            tuple: (predicted_class_index, class_probabilities)
        """
        pass
    
    def predict_batch(self, face_images):
        """
        Predict ethnicity for a batch of face images
        
        Args:
            face_images: List of face images
            
        Returns:
            list: List of (predicted_class_index, class_probabilities) tuples
        """
        return [self.predict(face) for face in face_images]
    
    def predict_top_n(self, face_image, n=3):
        """
        Predict top N ethnicities for a face image
        
        Args:
            face_image: Face image (already cropped and aligned)
            n: Number of top predictions to return
            
        Returns:
            list: List of (class_index, probability) tuples, sorted by probability
        """
        _, probabilities = self.predict(face_image)
        
        # Get indices sorted by probability (descending)
        indices = np.argsort(probabilities)[::-1]
        
        # Return top N
        top_n_indices = indices[:n]
        top_n_probs = probabilities[top_n_indices]
        
        return list(zip(top_n_indices, top_n_probs))