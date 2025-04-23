import os
import sys

# Tambahkan direktori project root ke Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Sekarang import
from src.config import (
    PROCESSED_DATA_DIR, 
    ETHNICITY_MODEL_DIR,
    BATCH_SIZE, 
    NUM_EPOCHS, 
    LEARNING_RATE, 
    RANDOM_SEED
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from src.data.dataset import FaceDataset
from src.data.augmentation import FaceAugmentation
from src.ethnicity_detection.cnn_classifier import CNNEthnicityModel

# Sisanya sama seperti script sebelumnya
def train_ethnicity_model():
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Cek ketersediaan GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan perangkat: {device}")
    
    # Path dataset
    train_csv = os.path.join(PROCESSED_DATA_DIR, 'splits', 'train.csv')
    val_csv = os.path.join(PROCESSED_DATA_DIR, 'splits', 'val.csv')
    
    # Buat dataset
    train_dataset = FaceDataset(
        train_csv, 
        PROCESSED_DATA_DIR, 
        transform=FaceAugmentation.get_train_transforms()
    )
    
    val_dataset = FaceDataset(
        val_csv, 
        PROCESSED_DATA_DIR, 
        transform=FaceAugmentation.get_val_transforms()
    )
    
    # DataLoader
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
    
    # Hitung jumlah kelas
    num_classes = len(set(train_dataset.face_frame['ethnicity']))
    
    # Inisialisasi model
    model = CNNEthnicityModel(num_classes=num_classes)
    model = model.to(device)
    
    # Loss dan optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Proses training
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['ethnicity'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validasi
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['ethnicity'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Cetak metrik
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Akurasi: {100*train_correct/train_total:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Akurasi: {100*val_correct/val_total:.2f}%")
        
        # Update learning rate
        scheduler.step(val_loss/len(val_loader))
    
    # Simpan model
    os.makedirs(ETHNICITY_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(ETHNICITY_MODEL_DIR, 'ethnicity_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model disimpan di {model_path}")

def main():
    try:
        train_ethnicity_model()
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()