"""
Split data into train/validation/test sets.

This script:
1. Reads images from data/raw/
2. Splits them 80/10/10 (train/val/test)
3. Copies to data/processed/ maintaining folder structure
4. Uses stratified split (balanced classes)
"""

import os
import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random

# Set random seed for reproducibility
random.seed(42)


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params


def get_image_paths(raw_data_dir):
    """
    Get all image paths organized by emotion class.
    
    Returns:
        dict: {emotion: [list of image paths]}
    """
    image_paths = defaultdict(list)
    
    # Supported image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.dng', '.DNG']
    
    # Iterate through each emotion folder
    for emotion_folder in Path(raw_data_dir).iterdir():
        if emotion_folder.is_dir():
            emotion = emotion_folder.name
            
            # Get all images in this emotion folder
            for img_path in emotion_folder.iterdir():
                if img_path.suffix in extensions:
                    image_paths[emotion].append(str(img_path))
    
    return image_paths


def split_data(image_paths, train_size, val_size, test_size, random_state):
    """
    Split image paths into train/val/test sets.
    
    Args:
        image_paths: dict of {emotion: [image_paths]}
        train_size: float (e.g., 0.8 for 80%)
        val_size: float
        test_size: float
        random_state: int for reproducibility
    
    Returns:
        tuple: (train_paths, val_paths, test_paths)
    """
    train_paths = defaultdict(list)
    val_paths = defaultdict(list)
    test_paths = defaultdict(list)
    
    # Split each emotion separately (stratified split)
    for emotion, paths in image_paths.items():
        # First split: separate test set
        train_val_paths, test_paths_emotion = train_test_split(
            paths,
            test_size=test_size,
            random_state=random_state
        )
        
        # Second split: separate train and validation
        # Adjust validation size relative to remaining data
        val_size_adjusted = val_size / (train_size + val_size)
        train_paths_emotion, val_paths_emotion = train_test_split(
            train_val_paths,
            test_size=val_size_adjusted,
            random_state=random_state
        )
        
        # Store results
        train_paths[emotion] = train_paths_emotion
        val_paths[emotion] = val_paths_emotion
        test_paths[emotion] = test_paths_emotion
        
        # Print split info
        print(f"\n{emotion}:")
        print(f"  Total: {len(paths)}")
        print(f"  Train: {len(train_paths_emotion)}")
        print(f"  Val:   {len(val_paths_emotion)}")
        print(f"  Test:  {len(test_paths_emotion)}")
    
    return train_paths, val_paths, test_paths


def copy_files(split_paths, processed_data_dir, split_name):
    """
    Copy files to processed data directory.
    
    Args:
        split_paths: dict of {emotion: [image_paths]}
        processed_data_dir: base processed directory
        split_name: 'train', 'val', or 'test'
    """
    total_copied = 0
    
    for emotion, paths in split_paths.items():
        # Create destination directory
        dest_dir = Path(processed_data_dir) / split_name / emotion
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy each file
        for src_path in paths:
            src_file = Path(src_path)
            dest_file = dest_dir / src_file.name
            shutil.copy2(src_path, dest_file)
            total_copied += 1
    
    print(f"\n✅ Copied {total_copied} images to {split_name} set")


def main():
    """Main function to split data"""
    print("=" * 60)
    print("SPLITTING DATA INTO TRAIN/VALIDATION/TEST SETS")
    print("=" * 60)
    
    # Load parameters
    params = load_params()
    
    # Get paths
    raw_data_dir = params['paths']['raw_data']
    processed_data_dir = params['paths']['processed_data']
    
    # Get split ratios
    train_size = params['data']['train_size']
    val_size = params['data']['val_size']
    test_size = params['data']['test_size']
    random_state = params['data']['random_state']
    
    print(f"\n📂 Reading images from: {raw_data_dir}")
    print(f"📊 Split ratios: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Get all image paths
    image_paths = get_image_paths(raw_data_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ORIGINAL DATA DISTRIBUTION")
    print("=" * 60)
    total_images = 0
    for emotion, paths in image_paths.items():
        print(f"{emotion}: {len(paths)} images")
        total_images += len(paths)
    print(f"\nTotal: {total_images} images")
    
    # Split data
    print("\n" + "=" * 60)
    print("SPLITTING DATA")
    print("=" * 60)
    train_paths, val_paths, test_paths = split_data(
        image_paths, train_size, val_size, test_size, random_state
    )
    
    # Copy files to processed directory
    print("\n" + "=" * 60)
    print("COPYING FILES TO PROCESSED DIRECTORY")
    print("=" * 60)
    
    copy_files(train_paths, processed_data_dir, 'train')
    copy_files(val_paths, processed_data_dir, 'val')
    copy_files(test_paths, processed_data_dir, 'test')
    
    print("\n" + "=" * 60)
    print(" DATA SPLITTING COMPLETE!")
    print("=" * 60)
    print(f"\n Processed data saved to: {processed_data_dir}")
    


if __name__ == "__main__":
    main()

