"""
Data augmentation script for emotion detection
Creates multiple augmented versions of training images
"""

import os
import yaml
import cv2
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def augment_image(image, aug_params):
    """
    Apply random augmentation to an image
    
    Args:
        image: Input image (numpy array, BGR format, 0-255 range)
        aug_params: Augmentation parameters from params.yaml
    
    Returns:
        Augmented image (numpy array, BGR format, 0-255 range)
    """
    img = image.copy().astype(np.float32)  # Convert to float for operations
    h, w = img.shape[:2]
    
    # Random horizontal flip (if enabled)
    if aug_params.get('horizontal_flip', False):
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
    
    # Random rotation (if enabled)
    if aug_params.get('rotation_range', 0) > 0:
        angle = np.random.uniform(-aug_params['rotation_range'], 
                                 aug_params['rotation_range'])
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), 
                            borderMode=cv2.BORDER_REFLECT_101,
                            flags=cv2.INTER_LINEAR)
    
    # Random brightness adjustment (if enabled)
    if aug_params.get('brightness_range', 0) > 0:
        brightness_delta = np.random.uniform(
            -aug_params['brightness_range'] * 50,  # Scale to reasonable range
            aug_params['brightness_range'] * 50
        )
        img = img + brightness_delta
    
    # Random contrast adjustment (if enabled)
    if aug_params.get('contrast_range', 0) > 0:
        contrast_factor = np.random.uniform(
            1.0 - aug_params['contrast_range'],
            1.0 + aug_params['contrast_range']
        )
        mean = img.mean()
        img = (img - mean) * contrast_factor + mean
    
    # Clip values to valid range [0, 255]
    img = np.clip(img, 0, 255)
    
    # Random zoom (crop and resize) (if enabled)
    zoom_range = aug_params.get('zoom_range', 0)
    if zoom_range > 0 and np.random.random() > 0.5:
        zoom = np.random.uniform(1.0 - zoom_range, 1.0 + zoom_range)
        
        if zoom < 1.0:  # Zoom out (crop)
            new_h, new_w = int(h * zoom), int(w * zoom)
            
            if new_h > 10 and new_w > 10:  # Ensure valid dimensions
                # Calculate crop coordinates (center crop)
                top = (h - new_h) // 2
                left = (w - new_w) // 2
                
                # Crop
                img = img[top:top+new_h, left:left+new_w]
                # Resize back to original size
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        elif zoom > 1.0:  # Zoom in (crop larger area and resize)
            new_h, new_w = int(h / zoom), int(w / zoom)
            
            if new_h > 10 and new_w > 10 and new_h < h and new_w < w:
                # Calculate crop coordinates (center crop)
                top = (h - new_h) // 2
                left = (w - new_w) // 2
                
                # Crop
                img = img[top:top+new_h, left:left+new_w]
                # Resize back to original size
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Convert back to uint8
    img = img.astype(np.uint8)
    
    return img


def augment_emotion_folder(emotion_dir, num_augmentations, aug_params):
    """
    Augment all images in an emotion folder
    
    Args:
        emotion_dir: Path to emotion directory
        num_augmentations: Number of augmented versions to create per image
        aug_params: Augmentation parameters
    
    Returns:
        Number of images created
    """
    # Get all image files (exclude already augmented ones)
    image_files = [f for f in os.listdir(emotion_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                   and '_aug' not in f]  # Skip already augmented images
    
    if not image_files:
        logger.warning(f"No images found in {emotion_dir}")
        return 0
    
    logger.info(f"Augmenting {len(image_files)} images...")
    
    total_created = 0
    errors = 0
    
    for img_file in tqdm(image_files, desc=f"Processing {os.path.basename(emotion_dir)}"):
        img_path = os.path.join(emotion_dir, img_file)
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Could not read {img_path}")
            errors += 1
            continue
        
        # Verify image is not empty
        if img.size == 0:
            logger.warning(f"Empty image: {img_path}")
            errors += 1
            continue
        
        # Create augmented versions
        base_name = os.path.splitext(img_file)[0]
        ext = os.path.splitext(img_file)[1]
        
        for i in range(1, num_augmentations + 1):
            try:
                # Apply augmentation
                aug_img = augment_image(img, aug_params)
                
                # Verify augmented image is valid
                if aug_img is None or aug_img.size == 0:
                    logger.warning(f"Invalid augmented image for {img_file}, iteration {i}")
                    errors += 1
                    continue
                
                # Save augmented image
                aug_filename = f"{base_name}_aug{i}{ext}"
                aug_path = os.path.join(emotion_dir, aug_filename)
                
                success = cv2.imwrite(aug_path, aug_img)
                
                if success:
                    total_created += 1
                else:
                    logger.warning(f"Failed to write {aug_path}")
                    errors += 1
                    
            except Exception as e:
                logger.error(f"Error augmenting {img_file}, iteration {i}: {str(e)}")
                errors += 1
                continue
    
    if errors > 0:
        logger.warning(f"Encountered {errors} errors during augmentation")
    
    return total_created


def main():
    """Main augmentation pipeline"""
    logger.info("=" * 60)
    logger.info("Starting data augmentation pipeline...")
    logger.info("=" * 60)
    
    # Load parameters
    params = load_params()
    aug_params = params['preprocessing']['augmentation']
    num_augmentations = params['preprocessing'].get('num_augmentations', 10)
    
    logger.info(f"Number of augmentations per image: {num_augmentations}")
    logger.info(f"Augmentation parameters:")
    for key, value in aug_params.items():
        logger.info(f"  - {key}: {value}")
    
    # Define train directory
    train_dir = "data/processed/train"
    
    if not os.path.exists(train_dir):
        logger.error(f"Training directory not found: {train_dir}")
        return
    
    # Get all emotion subdirectories
    emotion_dirs = [d for d in os.listdir(train_dir) 
                   if os.path.isdir(os.path.join(train_dir, d))]
    
    logger.info(f"\nFound emotions: {emotion_dirs}")
    
    total_augmented = 0
    
    # Augment each emotion folder
    for emotion in emotion_dirs:
        logger.info("\n" + "=" * 60)
        logger.info(f"Processing {emotion.upper()} emotion...")
        logger.info("=" * 60)
        
        emotion_path = os.path.join(train_dir, emotion)
        count = augment_emotion_folder(emotion_path, num_augmentations, aug_params)
        
        total_augmented += count
        logger.info(f"✓ Created {count} augmented images for {emotion}")
    
    # Count final images
    final_count = len([f for f in Path(train_dir).rglob('*') 
                      if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✓ AUGMENTATION COMPLETE!")
    logger.info(f"Original images: 276")
    logger.info(f"Augmented images created: {total_augmented}")
    logger.info(f"Total training images: {final_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()