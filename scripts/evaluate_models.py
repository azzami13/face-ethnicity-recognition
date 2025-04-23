import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)

from src.face_detection.mtcnn_detector import MTCNNDetector
from src.face_similarity.facenet_embedder import FaceNetEmbedder
from src.ethnicity_detection.cnn_classifier import CNNEthnicityClassifier
from src.config import PROCESSED_DATA_DIR, ETHNICITY_MAPPING_REVERSE

def evaluate_face_similarity(embedder, test_pairs_csv):
    """
    Evaluate face similarity model
    
    Args:
        embedder: Face embedding model
        test_pairs_csv: CSV file with face pairs for testing
    
    Returns:
        dict: Evaluation metrics
    """
    # Load test pairs
    df = pd.read_csv(test_pairs_csv)
    
    # Prepare lists to store results
    true_labels = []
    predicted_labels = []
    similarities = []
    
    # Iterate through pairs
    for _, row in df.iterrows():
        img1_path = os.path.join(PROCESSED_DATA_DIR, row['image1'])
        img2_path = os.path.join(PROCESSED_DATA_DIR, row['image2'])
        
        # Load images
        img1 = plt.imread(img1_path)
        img2 = plt.imread(img2_path)
        
        # Compare faces
        similarity = embedder.compare_faces(img1, img2)
        
        # Predict label based on a threshold
        predicted_label = 1 if similarity >= 0.6 else 0
        
        true_labels.append(row['same'])
        predicted_labels.append(predicted_label)
        similarities.append(similarity)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, predicted_labels),
        'precision': precision_score(true_labels, predicted_labels),
        'recall': recall_score(true_labels, predicted_labels),
        'f1_score': f1_score(true_labels, predicted_labels)
    }
    
    return metrics, true_labels, predicted_labels, similarities

def evaluate_ethnicity_detection(classifier, test_csv):
    """
    Evaluate ethnicity detection model
    
    Args:
        classifier: Ethnicity classification model
        test_csv: CSV file with test data
    
    Returns:
        dict: Evaluation metrics
    """
    # Load test data
    df = pd.read_csv(test_csv)
    
    # Prepare lists to store results
    true_labels = []
    predicted_labels = []
    
    # Detect faces and classify
    for _, row in df.iterrows():
        img_path = os.path.join(PROCESSED_DATA_DIR, row['image_path'])
        
        # Load image
        img = plt.imread(img_path)
        
        # Detect and classify
        detector = MTCNNDetector()
        faces = detector.extract_faces(img)
        
        if faces:
            # Use first detected face
            face = faces[0]['face']
            predicted_label, _ = classifier.predict(face)
            
            true_labels.append(row['ethnicity'])
            predicted_labels.append(predicted_label)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, predicted_labels),
        'precision': precision_score(true_labels, predicted_labels, average='weighted'),
        'recall': recall_score(true_labels, predicted_labels, average='weighted'),
        'f1_score': f1_score(true_labels, predicted_labels, average='weighted')
    }
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Classification report
    class_report = classification_report(
        true_labels, 
        predicted_labels, 
        target_names=[ETHNICITY_MAPPING_REVERSE[i] for i in range(len(ETHNICITY_MAPPING_REVERSE))]
    )
    
    return metrics, cm, class_report

def visualize_results(true_labels, predicted_labels, similarities=None):
    """
    Visualize evaluation results
    
    Args:
        true_labels: True labels
        predicted_labels: Predicted labels
        similarities: Similarity scores (optional)
    """
    # ROC Curve
    if similarities is not None:
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(true_labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

def main():
    # Load models
    face_embedder = FaceNetEmbedder()
    ethnicity_classifier = CNNEthnicityClassifier()
    
    # Paths to test data
    test_pairs_csv = os.path.join(PROCESSED_DATA_DIR, 'splits', 'test_pairs.csv')
    test_ethnicity_csv = os.path.join(PROCESSED_DATA_DIR, 'splits', 'test.csv')
    
    # Evaluate face similarity
    face_sim_metrics, true_labels, pred_labels, similarities = evaluate_face_similarity(
        face_embedder, test_pairs_csv
    )
    print("Face Similarity Metrics:")
    for metric, value in face_sim_metrics.items():
        print(f"{metric}: {value}")
    
    # Visualize face similarity results
    visualize_results(true_labels, pred_labels, similarities)
    
    # Evaluate ethnicity detection
    ethnicity_metrics, cm, class_report = evaluate_ethnicity_detection(
        ethnicity_classifier, test_ethnicity_csv
    )
    print("\nEthnicity Detection Metrics:")
    for metric, value in ethnicity_metrics.items():
        print(f"{metric}: {value}")
    
    print("\nClassification Report:")
    print(class_report)
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

if __name__ == "__main__":
    main()