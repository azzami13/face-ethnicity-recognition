import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, FACES_DIR, ALIGNED_FACES_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, FACE_DETECTION_MARGIN,
    ETHNICITY_MAPPING
)
from src.face_detection.mtcnn_detector import MTCNNDetector

def create_dataset_csv():
    """
    Create CSV file with image paths and labels
    Adapted for new directory structure: /raw/[ethnicity]/[name]/images
    """
    data = []
    
    # Traverse the raw data directory
    for ethnicity in os.listdir(RAW_DATA_DIR):
        ethnicity_dir = os.path.join(RAW_DATA_DIR, ethnicity)
        
        # Pastikan ini adalah direktori (bukan file)
        if os.path.isdir(ethnicity_dir):
            for name in os.listdir(ethnicity_dir):
                name_dir = os.path.join(ethnicity_dir, name)
                
                # Pastikan ini adalah direktori nama
                if os.path.isdir(name_dir):
                    for img_file in os.listdir(name_dir):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            # Konstruksi path relatif
                            img_path = os.path.join(ethnicity, name, img_file)
                            
                            data.append({
                                'image_path': img_path,
                                'name': name,
                                'ethnicity': ethnicity,
                            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Simpan ke CSV
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'labels.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Dataset CSV created at {csv_path}")
    
    return df

def detect_and_crop_faces():
    """
    Detect faces in raw images and crop them
    Updated to work with new directory structure
    """
    # Load CSV
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'labels.csv')
    if not os.path.exists(csv_path):
        df = create_dataset_csv()
    else:
        df = pd.read_csv(csv_path)
    
    # Initialize face detector
    detector = MTCNNDetector()
    
    # Process each image
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Detecting and cropping faces"):
        # Load image
        img_path = os.path.join(RAW_DATA_DIR, row['image_path'])
        
        # Pastikan file ada
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        try:
            # Baca gambar
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to read image: {img_path}")
                continue
            
            # Convert to RGB (OpenCV loads as BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = detector.extract_faces(img, padding=FACE_DETECTION_MARGIN)
            
            if not faces:
                print(f"Warning: No face detected in {img_path}")
                continue
            
            # Gunakan wajah dengan confidence tertinggi
            face_data = max(faces, key=lambda x: x['confidence'])
            face_img = face_data['face']
            
            # Buat direktori output
            output_dir = os.path.join(FACES_DIR, os.path.dirname(row['image_path']))
            os.makedirs(output_dir, exist_ok=True)
            
            # Simpan wajah yang di-crop
            output_path = os.path.join(FACES_DIR, row['image_path'])
            cv2.imwrite(output_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"Face detection and cropping completed. Faces saved to {FACES_DIR}")

def split_dataset():
    """
    Split dataset into train, validation, and test sets
    Updated to ensure no person appears in multiple splits
    """
    # Load CSV
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'labels.csv')
    if not os.path.exists(csv_path):
        print("Error: labels.csv not found. Run create_dataset_csv() first.")
        return
    
    df = pd.read_csv(csv_path)
    
    # Check if faces have been extracted
    sample_face_path = os.path.join(FACES_DIR, df.iloc[0]['image_path'])
    if not os.path.exists(sample_face_path):
        print("Error: Faces not detected. Run detect_and_crop_faces() first.")
        return
    
    # Update image paths to point to cropped faces
    df['image_path'] = df['image_path'].apply(lambda x: os.path.join('faces', x))
    
    # Group by name to ensure same person doesn't appear in different splits
    grouped = df.groupby('name')
    names = list(grouped.groups.keys())
    
    # Split names into train, val, test
    train_names, temp_names = train_test_split(
        names, train_size=TRAIN_RATIO, random_state=42
    )
    
    val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_names, test_names = train_test_split(
        temp_names, train_size=val_size, random_state=42
    )
    
    # Create DataFrames for each split
    train_df = df[df['name'].isin(train_names)]
    val_df = df[df['name'].isin(val_names)]
    test_df = df[df['name'].isin(test_names)]
    
    # Save splits
    splits_dir = os.path.join(PROCESSED_DATA_DIR, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(splits_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(splits_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(splits_dir, 'test.csv'), index=False)
    
    print(f"Dataset split completed: "
          f"Train: {len(train_df)} samples, "
          f"Validation: {len(val_df)} samples, "
          f"Test: {len(test_df)} samples")

def prepare_pair_datasets(df, output_path, num_pairs_per_person=10):
    """
    Buat dataset pasangan untuk face similarity
    
    Args:
        df: DataFrame dengan daftar gambar
        output_path: Path penyimpanan file CSV pasangan
        num_pairs_per_person: Jumlah pasangan per orang
    """
    # Kelompokkan berdasarkan nama
    grouped = df.groupby('name')
    pairs = []
    
    # Buat pasangan positif (orang yang sama)
    for name, group in grouped:
        # Kombinasi gambar untuk pasangan positif
        positive_pairs = group.sample(n=min(len(group), num_pairs_per_person * 2), replace=True)
        
        for i in range(0, len(positive_pairs), 2):
            if i + 1 < len(positive_pairs):
                pairs.append({
                    'image1': positive_pairs.iloc[i]['image_path'],
                    'image2': positive_pairs.iloc[i+1]['image_path'],
                    'same': 1
                })
    
    # Buat pasangan negatif (orang berbeda)
    names = list(grouped.groups.keys())
    for _ in range(len(pairs)):
        # Pilih dua nama berbeda
        name1, name2 = np.random.choice(names, 2, replace=False)
        
        # Ambil gambar acak dari kedua nama
        img1 = grouped.get_group(name1).sample(1).iloc[0]['image_path']
        img2 = grouped.get_group(name2).sample(1).iloc[0]['image_path']
        
        pairs.append({
            'image1': img1,
            'image2': img2,
            'same': 0
        })
    
    # Buat DataFrame dan simpan
    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(output_path, index=False)
    print(f"Pasangan gambar disimpan di {output_path}")

def prepare_dataset():
    """
    Jalankan seluruh pipeline persiapan dataset
    """
    # Buat dataset CSV
    df = create_dataset_csv()
    
    # Deteksi dan crop wajah
    detect_and_crop_faces()
    
    # Bagi dataset
    split_dataset()
    
    # Buat dataset pasangan untuk face similarity
    splits_dir = os.path.join(PROCESSED_DATA_DIR, 'splits')
    
    # Dataset pasangan untuk training
    prepare_pair_datasets(
        pd.read_csv(os.path.join(splits_dir, 'train.csv')),
        os.path.join(splits_dir, 'train_pairs.csv')
    )
    
    # Dataset pasangan untuk validasi
    prepare_pair_datasets(
        pd.read_csv(os.path.join(splits_dir, 'val.csv')),
        os.path.join(splits_dir, 'val_pairs.csv')
    )
    
    # Dataset pasangan untuk testing
    prepare_pair_datasets(
        pd.read_csv(os.path.join(splits_dir, 'test.csv')),
        os.path.join(splits_dir, 'test_pairs.csv')
    )
    
    print("Dataset preparation completed!")

if __name__ == "__main__":
    prepare_dataset()