import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from src.ethnicity_detection.classifier import EthnicityClassifier
from src.config import ETHNICITY_MODEL_DIR, ETHNICITY_MAPPING_REVERSE

class CNNEthnicityModel(nn.Module):
    """CNN model for ethnicity classification using transfer learning"""
    
    def __init__(self, num_classes=3):
        super(CNNEthnicityModel, self).__init__()
        
        # Load pre-trained ResNet model
        self.base_model = models.resnet50(pretrained=True)
        
        # Replace the last fully connected layer
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)


class CNNEthnicityClassifier(EthnicityClassifier):
    """Ethnicity classifier using CNN with transfer learning"""
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize ethnicity classifier
        
        Args:
            model_path: Path to trained model
            device: Device to run the model on ('cpu' or 'cuda')
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Set model path
        if model_path is None:
            model_path = os.path.join(ETHNICITY_MODEL_DIR, "ethnicity_model.pth")
        
        # Initialize model
        self.model = CNNEthnicityModel()
        
        # Load trained weights if available
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded ethnicity model from {model_path}")
        else:
            print(f"Warning: Ethnicity model not found at {model_path}")
            print("Using untrained model")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        """
        Preprocess image for the model
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if image.shape[2] == 3:  # Check if color image
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict(self, face_image):
        """
        Predict ethnicity for a face image
        
        Args:
            face_image: Face image (already cropped)
            
        Returns:
            tuple: (predicted_class_index, class_probabilities)
        """
        # Preprocess image
        tensor = self.preprocess_image(face_image).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Get predicted class
        predicted_class = np.argmax(probabilities)
        
        return predicted_class, probabilities