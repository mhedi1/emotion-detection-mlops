"""
Prepare FER2013 dataset - subsample from folder structure
Optimized for 6GB GPU
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_fer_structure():
    """Create directory structure for FER2013 subset"""
    base_dir = Path("data/fer2013_subset")
    splits = ['train', 'validation', 'test']
    emotions = ['angry', 'happy', 'neutral', 'sad', 'surprise']
    
    for split in splits:
        for emotion in emotions:
            path = base_dir / split / emotion
            path.mkdir(parents=True, exist_ok=True)
    
    logger.info("✓ Created FER2013 subset structure")


def process_folder_structure(source_dir, output_base, samples_per_emotion=2000):
    """
    Process FER2013 from folder structure with intelligent subsampling
    
    Args:
        source_dir: Path to FER2013 with train/test folders
        output_base: Output directory for subset
        samples_per_emotion: Max samples per emotion for training
    """
    logger.info("="*60)
    logger.info("Processing FER2013 from Folder Structure")
    logger.info("="*60)
    
    source_path = Path(source_dir)
    
    # Check structure
    train_dir = source_path / "train"
    test_dir = source_path / "test"
    
    if not train_dir.exists():
        logger.error(f"❌ Train directory not found: {train_dir}")
        return 0
    
    logger.info(f"✓ Found train directory: {train_dir}")
    if test_dir.exists():
        logger.info(f"✓ Found test directory: {test_dir}")
    else:
        logger.warning(f"⚠️  Test directory not found, will split from train")
    
    # Our emotion mapping
    emotions = ['angry', 'happy', 'neutral', 'sad', 'surprise']
    
    # Count statistics
    emotion_counts = {
        'train': {emotion: 0 for emotion in emotions},
        'validation': {emotion: 0 for emotion in emotions},
        'test': {emotion: 0 for emotion in emotions}
    }
    
    # Target counts per split
    target_counts = {
        'train': samples_per_emotion,           # 2000 per emotion
        'validation': samples_per_emotion // 5, # 400 per emotion
        'test': samples_per_emotion // 5        # 400 per emotion
    }
    
    total_processed = 0
    
    # Process each emotion
    for emotion in emotions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {emotion.upper()}")
        logger.info(f"{'='*60}")
        
        # Collect all images for this emotion
        train_emotion_dir = train_dir / emotion
        test_emotion_dir = test_dir / emotion if test_dir.exists() else None
        
        # Get training images
        train_images = []
        if train_emotion_dir.exists():
            train_images = list(train_emotion_dir.glob("*.jpg")) + \
                          list(train_emotion_dir.glob("*.jpeg")) + \
                          list(train_emotion_dir.glob("*.png"))
            logger.info(f"Found {len(train_images)} images in train/{emotion}")
        else:
            logger.warning(f"⚠️  No train/{emotion} directory found")
            continue
        
        # Get test images
        test_images = []
        if test_emotion_dir and test_emotion_dir.exists():
            test_images = list(test_emotion_dir.glob("*.jpg")) + \
                         list(test_emotion_dir.glob("*.jpeg")) + \
                         list(test_emotion_dir.glob("*.png"))
            logger.info(f"Found {len(test_images)} images in test/{emotion}")
        
        # Shuffle for random sampling
        random.seed(42)
        random.shuffle(train_images)
        random.shuffle(test_images)
        
        # === TRAIN SPLIT ===
        train_needed = target_counts['train']
        train_selected = train_images[:min(train_needed, len(train_images))]
        
        for img_file in tqdm(train_selected, desc=f"Train {emotion}"):
            try:
                # Read image
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                # Resize to 224x224 if needed
                if img.shape[:2] != (224, 224):
                    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                
                # Convert grayscale to RGB if needed
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
                # Save to subset
                dest_path = output_base / 'train' / emotion / f"fer_{emotion}_{img_file.stem}.jpg"
                cv2.imwrite(str(dest_path), img)
                
                emotion_counts['train'][emotion] += 1
                total_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
                continue
        
        # === VALIDATION SPLIT ===
        val_needed = target_counts['validation']
        
        if len(train_images) > train_needed:
            # Use remaining train images
            val_selected = train_images[train_needed:train_needed + val_needed]
        else:
            # Use test images
            val_selected = test_images[:min(val_needed, len(test_images))]
        
        for img_file in tqdm(val_selected, desc=f"Val {emotion}"):
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                if img.shape[:2] != (224, 224):
                    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
                dest_path = output_base / 'validation' / emotion / f"fer_{emotion}_{img_file.stem}.jpg"
                cv2.imwrite(str(dest_path), img)
                
                emotion_counts['validation'][emotion] += 1
                total_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
                continue
        
        # === TEST SPLIT ===
        test_needed = target_counts['test']
        
        if test_images:
            # Start after validation samples
            test_start = val_needed if len(train_images) <= train_needed else 0
            test_selected = test_images[test_start:test_start + min(test_needed, len(test_images) - test_start)]
        else:
            # Use remaining train images
            test_start = train_needed + val_needed
            test_selected = train_images[test_start:test_start + min(test_needed, len(train_images) - test_start)]
        
        for img_file in tqdm(test_selected, desc=f"Test {emotion}"):
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                if img.shape[:2] != (224, 224):
                    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
                dest_path = output_base / 'test' / emotion / f"fer_{emotion}_{img_file.stem}.jpg"
                cv2.imwrite(str(dest_path), img)
                
                emotion_counts['test'][emotion] += 1
                total_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing {img_file}: {e}")
                continue
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("FER2013 Subset Summary")
    logger.info("="*60)
    
    for split in ['train', 'validation', 'test']:
        logger.info(f"\n{split.upper()}:")
        total = 0
        for emotion in sorted(emotions):
            count = emotion_counts[split][emotion]
            logger.info(f"  {emotion:10s}: {count:4d} images")
            total += count
        logger.info(f"  {'TOTAL':10s}: {total:4d} images")
    
    grand_total = sum(sum(counts.values()) for counts in emotion_counts.values())
    logger.info(f"\n✓ Total processed: {grand_total} images")
    
    return grand_total


def main():
    """Main preparation pipeline"""
    logger.info("="*60)
    logger.info("FER2013 PREPARATION FOR 6GB GPU")
    logger.info("="*60)
    
    # Check for FER2013 folder structure (UPDATED PATH)
    fer_dir = Path("data/external/fer2013")
    
    if not fer_dir.exists():
        logger.error(f"\n❌ FER2013 directory not found: {fer_dir}")
        logger.error("\nExpected structure:")
        logger.error("data/external/fer2013/")
        logger.error("├── train/")
        logger.error("│   ├── angry/")
        logger.error("│   ├── happy/")
        logger.error("│   └── ...")
        logger.error("└── test/")
        logger.error("    ├── angry/")
        logger.error("    └── ...")
        return
    
    logger.info(f"✓ Found FER2013 directory: {fer_dir}")
    
    # Create structure
    create_fer_structure()
    
    # Process folders
    output_base = Path("data/fer2013_subset")
    total = process_folder_structure(
        source_dir=fer_dir,
        output_base=output_base,
        samples_per_emotion=2000
    )
    
    if total == 0:
        logger.error("\n❌ No images were processed!")
        return
    
    logger.info("\n" + "="*60)
    logger.info("✓ FER2013 SUBSET READY FOR TRAINING!")
    logger.info(f"Location: {output_base.absolute()}")
    logger.info(f"Total images: {total}")
    logger.info("Perfect for RTX 3050 6GB!")
    logger.info("="*60)


if __name__ == "__main__":
    main()