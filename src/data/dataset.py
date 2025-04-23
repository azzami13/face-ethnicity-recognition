import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2

from src.config import ETHNICITY_MAPPING

class FaceDataset(Dataset):
    """Custom dataset for face data"""
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.face_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.face_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = os.path.join(self.root_dir, self.face_frame.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        
        name = self.face_frame.iloc[idx, 1]
        ethnicity = self.face_frame.iloc[idx, 2]
        
        # Convert ethnicity to numerical label
        ethnicity_label = ETHNICITY_MAPPING.get(ethnicity, -1)
        
        if ethnicity_label == -1:
            raise ValueError(f"Unknown ethnicity: {ethnicity}")
        
        sample = {'image': image, 'name': name, 'ethnicity': ethnicity_label}
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            
        return sample


class PairDataset(Dataset):
    """Dataset for generating pairs of face images for similarity training"""
    
    def __init__(self, csv_file, root_dir, transform=None, pairs_per_person=10):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            pairs_per_person (int): Number of positive and negative pairs per person.
        """
        self.face_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.pairs_per_person = pairs_per_person
        
        # Misalnya 'name' diambil dari folder orang, asumsinya struktur path: faces\Etnis\Nama\filename.jpg
        self.face_frame['name'] = self.face_frame['image1'].apply(lambda x: x.split('\\')[2])
        self.grouped = self.face_frame.groupby('name')
        self.people = list(self.grouped.groups.keys())

        self.pairs = self._generate_pairs()

        
    def _generate_pairs(self):
        """Generate positive and negative pairs for training"""
        pairs = []
        
        # For each person
        for person in self.people:
            person_indices = self.grouped.get_group(person).index.tolist()
            
            # Generate positive pairs (same person)
            if len(person_indices) >= 2:  # Need at least 2 images for positive pair
                for _ in range(min(self.pairs_per_person, len(person_indices))):
                    # Randomly select 2 different images of the same person
                    idx1, idx2 = np.random.choice(person_indices, 2, replace=False)
                    pairs.append((idx1, idx2, 1))  # 1 means same person
            
            # Generate negative pairs (different people)
            other_people = [p for p in self.people if p != person]
            if other_people:  # Need at least one other person for negative pair
                for _ in range(self.pairs_per_person):
                    # Randomly select an image of the current person
                    idx1 = np.random.choice(person_indices)
                    
                    # Randomly select another person
                    other_person = np.random.choice(other_people)
                    other_indices = self.grouped.get_group(other_person).index.tolist()
                    
                    # Randomly select an image of the other person
                    idx2 = np.random.choice(other_indices)
                    
                    pairs.append((idx1, idx2, 0))  # 0 means different people
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        idx1, idx2, same = self.pairs[idx]
        
        # Get first image
        img1_path = os.path.join(self.root_dir, self.face_frame.iloc[idx1, 0])
        image1 = Image.open(img1_path).convert('RGB')
        
        # Get second image
        img2_path = os.path.join(self.root_dir, self.face_frame.iloc[idx2, 0])
        image2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            
        return {'image1': image1, 'image2': image2, 'same': same}
        