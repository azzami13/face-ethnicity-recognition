import numpy as np
from scipy.spatial.distance import cosine, euclidean

def cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        float: Cosine similarity score
    """
    # Cosine similarity (1 is most similar, 0 is least similar)
    return 1 - cosine(embedding1, embedding2)

def euclidean_similarity(embedding1, embedding2):
    """
    Calculate Euclidean similarity between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        float: Euclidean similarity score
    """
    # Normalize distance to similarity score
    max_distance = np.sqrt(len(embedding1))  # Maximum possible Euclidean distance
    distance = euclidean(embedding1, embedding2)
    
    # Convert distance to similarity (closer = more similar)
    return 1 - (distance / max_distance)

def calculate_similarity(embedding1, embedding2, method='cosine'):
    """
    Calculate similarity between two face embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        method: Similarity calculation method ('cosine' or 'euclidean')
    
    Returns:
        float: Similarity score
    """
    if method == 'cosine':
        return cosine_similarity(embedding1, embedding2)
    elif method == 'euclidean':
        return euclidean_similarity(embedding1, embedding2)
    else:
        raise ValueError(f"Unsupported similarity method: {method}")

def verify_faces(embedding1, embedding2, threshold=0.6, method='cosine'):
    """
    Verify if two face embeddings represent the same person
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        threshold: Similarity threshold for verification
        method: Similarity calculation method
    
    Returns:
        tuple: (is_same_person, similarity_score)
    """
    similarity = calculate_similarity(embedding1, embedding2, method)
    is_same_person = similarity >= threshold
    
    return is_same_person, similarity