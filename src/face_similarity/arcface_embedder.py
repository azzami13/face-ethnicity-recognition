import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from src.face_similarity.embedder import FaceEmbedder
from src.config import FACE_EMBEDDING_MODEL_DIR, FACE_EMBEDDING_SIZE

class ArcFaceModel(nn.Module):
    """ArcFace embedding model architecture"""
    def __init__(self, embedding_size=FACE_EMBEDDING_SIZE):
        super().__init__()
        # Implementasi arsitektur ArcFace
        # Contoh sederhana, dalam praktiknya akan lebih kompleks
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc = nn.Linear(64 * 112 * 112, embedding_size)
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)

class ArcFaceEmbedder(FaceEmbedder):
    """Face embedder using ArcFace"""
    
    def __init__(self, model_path=None):
        """
        Initialize ArcFace embedder
        
        Args:
            model_path: Path to ArcFace model
        """
        if model_path is None:
            model_path = os.path.join(FACE_EMBEDDING_MODEL_DIR, "arcface_model.pth")
        
        # Initialize model
        self.model = ArcFaceModel()
        
        # Load model weights if available
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print(f"Loaded ArcFace model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"No pre-trained model found at {model_path}")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def preprocess_image(self, image):
        """
        Preprocess image for ArcFace model
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Ensure image is in RGB
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    def get_embedding(self, face_image):
        """
        Generate embedding vector for a face image
        
        Args:
            face_image: Face image (already cropped)
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        # Preprocess image
        tensor = self.preprocess_image(face_image)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model(tensor)
        
        # Convert to numpy
        return embedding.squeeze().numpy()