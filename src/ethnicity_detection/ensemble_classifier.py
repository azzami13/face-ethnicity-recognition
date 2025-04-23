import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler

from src.ethnicity_detection.classifier import EthnicityClassifier
from src.ethnicity_detection.cnn_classifier import CNNEthnicityModel
from src.config import ETHNICITY_MAPPING_REVERSE, PROCESSED_DATA_DIR

class EnsembleEthnicityClassifier(EthnicityClassifier):
    """
    Ensemble classifier for ethnicity detection
    Combines multiple models for improved prediction
    """
    
    def __init__(self, device=None):
        """
        Initialize ensemble classifier
        """
        # Set device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize base models
        self.base_models = [
            ('resnet', self._create_resnet_model()),
            ('vgg', self._create_vgg_model()),
            # Add more models as needed
        ]
        
        # Voting classifier
        self.ensemble = VotingClassifier(
            estimators=self.base_models,
            voting='soft'
        )
    
    def _create_resnet_model(self):
        """Create ResNet-based model"""
        model = CNNEthnicityModel(num_classes=len(ETHNICITY_MAPPING_REVERSE))
        model_path = f"{PROCESSED_DATA_DIR}/models/resnet_ethnicity_model.pth"
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception as e:
            print(f"Could not load ResNet model: {e}")
        
        return model
    
    def _create_vgg_model(self):
        """Create VGG-based model"""
        import torchvision.models as models
        
        class VGGEthnicityModel(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.base_model = models.vgg16(pretrained=True)
                num_features = self.base_model.classifier[6].in_features
                
                # Modify last layer
                features = list(self.base_model.classifier.children())[:-1]
                features.extend([
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                ])
                self.base_model.classifier = nn.Sequential(*features)
            
            def forward(self, x):
                return self.base_model(x)
        
        model = VGGEthnicityModel(num_classes=len(ETHNICITY_MAPPING_REVERSE))
        model_path = f"{PROCESSED_DATA_DIR}/models/vgg_ethnicity_model.pth"
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception as e:
            print(f"Could not load VGG model: {e}")
        
        return model
    
    def predict(self, face_image):
        """
        Predict ethnicity using ensemble method
        
        Args:
            face_image: Face image for prediction
            
        Returns:
            tuple: (predicted_class_index, class_probabilities)
        """
        # Preprocess image
        from src.data.augmentation import FaceAugmentation
        
        # Apply transforms
        transform = FaceAugmentation.get_val_transforms()
        
        # Convert image to tensor
        if isinstance(face_image, np.ndarray):
            from PIL import Image
            face_image = Image.fromarray(face_image)
        
        # Transform image
        image_tensor = transform(face_image).unsqueeze(0)
        
        # Ensemble prediction
        with torch.no_grad():
            # Collect predictions from base models
            predictions = []
            for _, model in self.base_models:
                model.eval()
                output = model(image_tensor)
                predictions.append(torch.softmax(output, dim=1).numpy())
            
            # Average probabilities
            avg_probabilities = np.mean(predictions, axis=0)[0]
        
        # Get predicted class
        predicted_class = np.argmax(avg_probabilities)
        
        return predicted_class, avg_probabilities

    def predict_top_n(self, face_image, n=3):
        """
        Predict top N ethnicities
        
        Args:
            face_image: Face image for prediction
            n: Number of top predictions
            
        Returns:
            List of (class_index, probability) tuples
        """
        _, probabilities = self.predict(face_image)
        
        # Get indices sorted by probability (descending)
        indices = np.argsort(probabilities)[::-1]
        
        # Return top N
        top_n_indices = indices[:n]
        top_n_probs = probabilities[top_n_indices]
        
        return list(zip(top_n_indices, top_n_probs))