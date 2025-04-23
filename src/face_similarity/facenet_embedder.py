import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.face_similarity.embedder import FaceEmbedder
from src.config import FACE_EMBEDDING_SIZE
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class FaceNetEmbedder(FaceEmbedder, nn.Module):
    """Face embedder using a dummy PyTorch model for compatibility"""

    def __init__(self, model_path=None):
        super(FaceNetEmbedder, self).__init__()

        print("Using placeholder model (PyTorch-based) for demonstration.")

        # Inisialisasi layer-layers dari model dummy
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((10, 10)),  # ⬅️ Untuk memastikan outputnya ukuran 64 x 10 x 10
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, FACE_EMBEDDING_SIZE)
        )

        # Transformasi gambar
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),  # Mengubah gambar menjadi tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def forward(self, x):
        return F.normalize(self.features(x), p=2, dim=1)

    def get_embedding(self, face_tensor):
        self.eval()
        
        with torch.no_grad():
            # Jika input berupa numpy array, ubah jadi PIL Image
            if isinstance(face_tensor, np.ndarray):
                face_tensor = Image.fromarray(face_tensor)
            
            # Terapkan transformasi untuk ubah gambar menjadi tensor
            face_tensor = self.transform(face_tensor)

            # Tambahkan dimensi batch ke tensor (torch membutuhkan batch dimensi)
            face_tensor = face_tensor.unsqueeze(0)

            # Dapatkan embedding dari wajah
            return self(face_tensor)[0]
