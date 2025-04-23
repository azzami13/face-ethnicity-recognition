import os
import json
import numpy as np
import pandas as pd
import cv2
import yaml

def save_json(data, filepath):
    """
    Simpan data ke file JSON
    
    Args:
        data: Data yang akan disimpan
        filepath: Path file tujuan
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(filepath):
    """
    Muat data dari file JSON
    
    Args:
        filepath: Path file sumber
    
    Returns:
        Data yang dimuat
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_numpy(array, filepath):
    """
    Simpan numpy array
    
    Args:
        array: Numpy array yang akan disimpan
        filepath: Path file tujuan
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.save(filepath, array)

def load_numpy(filepath):
    """
    Muat numpy array
    
    Args:
        filepath: Path file sumber
    
    Returns:
        Numpy array
    """
    return np.load(filepath)

def save_image(image, filepath):
    """
    Simpan gambar
    
    Args:
        image: Gambar (numpy array)
        filepath: Path file tujuan
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, image)

def load_image(filepath):
    """
    Muat gambar
    
    Args:
        filepath: Path file sumber
    
    Returns:
        Gambar dalam format numpy array
    """
    return cv2.imread(filepath)

def save_config(config, filepath):
    """
    Simpan konfigurasi ke file YAML
    
    Args:
        config: Dictionary konfigurasi
        filepath: Path file tujuan
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config(filepath):
    """
    Muat konfigurasi dari file YAML
    
    Args:
        filepath: Path file sumber
    
    Returns:
        Dictionary konfigurasi
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def list_files(directory, extensions=None):
    """
    Daftar file dalam direktori
    
    Args:
        directory: Direktori sumber
        extensions: Filter ekstensi file
    
    Returns:
        Daftar path file
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            if extensions is None or any(filepath.endswith(ext) for ext in extensions):
                files.append(filepath)
    return files