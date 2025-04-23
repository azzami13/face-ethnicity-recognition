import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    roc_curve, 
    auc
)

def calculate_classification_metrics(y_true, y_pred, labels=None):
    """
    Hitung metrik klasifikasi
    
    Args:
        y_true: Label sebenarnya
        y_pred: Label prediksi
        labels: Label kelas
    
    Returns:
        Dictionary metrik klasifikasi
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', labels=labels),
        'recall': recall_score(y_true, y_pred, average='weighted', labels=labels),
        'f1_score': f1_score(y_true, y_pred, average='weighted', labels=labels)
    }
    return metrics

def calculate_confusion_matrix(y_true, y_pred, labels=None):
    """
    Hitung matriks konfusi
    
    Args:
        y_true: Label sebenarnya
        y_pred: Label prediksi
        labels: Label kelas
    
    Returns:
        Matriks konfusi
    """
    return confusion_matrix(y_true, y_pred, labels=labels)

def calculate_roc_curve(y_true, y_scores):
    """
    Hitung kurva ROC
    
    Args:
        y_true: Label sebenarnya
        y_scores: Skor probabilitas
    
    Returns:
        Tuple (fpr, tpr, threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    return fpr, tpr, thresholds

def calculate_auc(y_true, y_scores):
    """
    Hitung Area Under Curve (AUC)
    
    Args:
        y_true: Label sebenarnya
        y_scores: Skor probabilitas
    
    Returns:
        Nilai AUC
    """
    return auc(calculate_roc_curve(y_true, y_scores)[0], 
               calculate_roc_curve(y_true, y_scores)[1])

def calculate_similarity_metrics(embeddings1, embeddings2, labels):
    """
    Hitung metrik similarity
    
    Args:
        embeddings1: Embedding set pertama
        embeddings2: Embedding set kedua
        labels: Label pasangan
    
    Returns:
        Dictionary metrik similarity
    """
    # Hitung jarak atau similaritas
    from scipy.spatial.distance import cosine
    
    # Hitung similaritas antar embedding
    similarities = []
    for emb1, emb2 in zip(embeddings1, embeddings2):
        similarities.append(1 - cosine(emb1, emb2))
    
    # Konversi ke numpy array
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    # Hitung metrik
    metrics = {
        'mean_similarity': np.mean(similarities),
        'std_similarity': np.std(similarities),
        'similarity_same_class': np.mean(similarities[labels == 1]),
        'similarity_diff_class': np.mean(similarities[labels == 0])
    }
    
    return metrics