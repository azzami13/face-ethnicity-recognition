import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.dataset import PairDataset
from src.data.augmentation import FaceAugmentation
from src.config import (
    PROCESSED_DATA_DIR, FACE_EMBEDDING_MODEL_DIR,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, RANDOM_SEED
)

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for face similarity learning
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        """
        Compute contrastive loss
        
        Args:
            output1: Embedding for first image
            output2: Embedding for second image
            label: Binary label (0 for different, 1 for same)
        """
        # Compute Euclidean distance
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        
        # Contrastive loss calculation
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss_contrastive

def train_face_similarity_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Train face similarity model using contrastive learning
    
    Args:
        model: Face embedding model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Training device (CPU/GPU)
    
    Returns:
        Trained model
    """
    # Initialize lists to track metrics
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # Get data
            img1 = batch['image1'].to(device)
            img2 = batch['image2'].to(device)
            labels = batch['same'].float().to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            output1 = model(img1)
            output2 = model(img2)
            
            # Compute loss
            loss = criterion(output1, output2, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Get data
                img1 = batch['image1'].to(device)
                img2 = batch['image2'].to(device)
                labels = batch['same'].float().to(device)
                
                # Forward pass
                output1 = model(img1)
                output2 = model(img2)
                
                # Compute loss
                loss = criterion(output1, output2, labels)
                val_loss += loss.item()
        
        # Average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(FACE_EMBEDDING_MODEL_DIR, 'best_face_similarity_model.pth')
            os.makedirs(FACE_EMBEDDING_MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with validation loss: {avg_val_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(FACE_EMBEDDING_MODEL_DIR, 'loss_curves.png'))
    plt.close()
    
    return model

def main():
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = PairDataset(
        os.path.join(PROCESSED_DATA_DIR, 'splits', 'train_pairs.csv'),
        PROCESSED_DATA_DIR,
        transform=FaceAugmentation.get_train_transforms()
    )
    
    val_dataset = PairDataset(
        os.path.join(PROCESSED_DATA_DIR, 'splits', 'val_pairs.csv'),
        PROCESSED_DATA_DIR,
        transform=FaceAugmentation.get_val_transforms()
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4
    )
    
    # Define model 
    from src.face_similarity.facenet_embedder import FaceNetEmbedder
    model = FaceNetEmbedder()
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    trained_model = train_face_similarity_model(
        model, train_loader, val_loader, 
        criterion, optimizer, NUM_EPOCHS, device
    )
    
    print("Face Similarity Model Training Completed.")

if __name__ == "__main__":
    main()