"""
Script for preparing the dataset

This script:
1. Creates a CSV file with all the raw images and their labels
2. Detects and crops faces from the raw images
3. Splits the dataset into train, validation, and test sets
"""
import os
import sys
import argparse

# Tambahkan direktori project root ke Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Sekarang import
from src.data.preprocessing import prepare_dataset
from src.config import RAW_DATA_DIR

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the dataset for face recognition")
    parser.add_argument('--data_dir', type=str, default=RAW_DATA_DIR,
                       help='Directory containing the raw images')
    return parser.parse_args()

def main():
    try:
        args = parse_args()
        
        # Check if raw data directory exists
        if not os.path.exists(args.data_dir):
            print(f"Error: Raw data directory not found: {args.data_dir}")
            print("Please create the directory and add your images first.")
            print(f"Current RAW_DATA_DIR: {args.data_dir}")
            return
        
        # Check if raw data directory has any data
        if not os.listdir(args.data_dir):
            print(f"Error: Raw data directory is empty: {args.data_dir}")
            print("Please add your images first, organized as /[ethnicity]/[name]/image.jpg")
            return
        
        # Prepare the dataset
        prepare_dataset()
        print("Dataset preparation completed!")
    
    except ImportError as e:
        print(f"Import error: {e}")
        print("Current sys.path:", sys.path)
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()