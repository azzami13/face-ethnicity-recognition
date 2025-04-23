import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance

class FaceAugmentation:
    """Class for face image augmentation"""
    
    @staticmethod
    def get_train_transforms():
        """Get transforms for training data"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_val_transforms():
        """Get transforms for validation/test data"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def apply_random_noise(image, noise_level=0.05):
        """Add random Gaussian noise to image"""
        if isinstance(image, torch.Tensor):
            # Convert to numpy if tensor
            was_tensor = True
            device = image.device
            if image.dim() == 4:  # batch of images
                image = image.cpu().numpy()
            else:
                image = image.unsqueeze(0).cpu().numpy()
        else:
            was_tensor = False
        
        # Add noise
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_image = np.clip(image + noise, 0, 1)
        
        if was_tensor:
            # Convert back to tensor
            noisy_image = torch.from_numpy(noisy_image).to(device)
            if image.shape[0] == 1:  # single image
                noisy_image = noisy_image.squeeze(0)
                
        return noisy_image

    @staticmethod
    def apply_random_occlusion(image, max_area=0.3):
        """Apply random rectangular occlusion"""
        if isinstance(image, torch.Tensor):
            return image  # Skip for tensors
            
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
            
        width, height = image.size
        
        # Random occlusion size
        occ_width = int(width * np.random.uniform(0.1, max_area))
        occ_height = int(height * np.random.uniform(0.1, max_area))
        
        # Random position
        x = np.random.randint(0, width - occ_width)
        y = np.random.randint(0, height - occ_height)
        
        # Create occlusion
        occluded_image = image.copy()
        occlusion = Image.new('RGB', (occ_width, occ_height), color=(0, 0, 0))
        occluded_image.paste(occlusion, (x, y))
        
        return occluded_image
    
    @staticmethod
    def apply_random_lighting(image, var=0.2):
        """Apply random lighting changes"""
        if isinstance(image, torch.Tensor) or isinstance(image, np.ndarray):
            return image  # Skip for tensors and numpy arrays
        
        # Random brightness
        factor = np.random.uniform(1-var, 1+var)
        image = ImageEnhance.Brightness(image).enhance(factor)
        
        # Random contrast
        factor = np.random.uniform(1-var, 1+var)
        image = ImageEnhance.Contrast(image).enhance(factor)
        
        return image