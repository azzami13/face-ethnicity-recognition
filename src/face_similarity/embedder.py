from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

class FaceEmbedder(ABC):
    """Abstract base class for face embedding models"""
    
    @abstractmethod
    def get_embedding(self, face_image):
        """
        Generate embedding vector for a face image
        
        Args:
            face_image: Face image (already cropped and aligned)
            
        Returns:
            np.ndarray: Embedding vector
        """
        pass
    
    def compare_faces(self, face1, face2, metric='cosine'):
        """
        Compare two face images and return similarity score
        
        Args:
            face1: First face image
            face2: Second face image
            metric: Similarity metric ('cosine' or 'euclidean')
            
        Returns:
            float: Similarity score (higher means more similar)
        """
        emb1 = self.get_embedding(face1)
        emb2 = self.get_embedding(face2)
        
        if metric == 'cosine':
            # Using sklearn's cosine_similarity
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            # Scale to [0, 1]
            similarity = (similarity + 1) / 2
        elif metric == 'euclidean':
            # Convert distance to similarity (inverse relationship)
            distance = euclidean(emb1, emb2)
            max_distance = np.sqrt(len(emb1) * 4)  # Theoretical max distance for normalized vectors
            similarity = 1 - (distance / max_distance)
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        return similarity
    
    def verify_faces(self, face1, face2, threshold, metric='cosine'):
        """
        Verify if two faces are of the same person
        
        Args:
            face1: First face image
            face2: Second face image
            threshold: Similarity threshold for verification
            metric: Similarity metric ('cosine' or 'euclidean')
            
        Returns:
            tuple: (is_same_person, similarity_score)
        """
        similarity = self.compare_faces(face1, face2, metric)
        is_same_person = similarity >= threshold
        
        return is_same_person, similarity