# src/data/convert_dng_to_jpg.py
"""
Convert DNG (RAW) files to JPG format for deep learning
"""
import rawpy
import imageio
from pathlib import Path
from tqdm import tqdm
import yaml

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def convert_dng_to_jpg(dng_path, jpg_path, quality=95):
    """
    Convert a single DNG file to JPG
    
    Args:
        dng_path: Path to input DNG file
        jpg_path: Path to output JPG file
        quality: JPG quality (0-100)
    """
    try:
        # Read DNG file
        with rawpy.imread(str(dng_path)) as raw:
            # Convert to RGB image
            rgb = raw.postprocess()
        
        # Save as JPG
        imageio.imwrite(str(jpg_path), rgb, quality=quality)
        return True
    
    except Exception as e:
        print(f"❌ Error converting {dng_path.name}: {e}")
        return False

def convert_all_dng_files(params):
    """
    Convert all DNG files in data/raw/ to JPG
    """
    print("🔄 Starting DNG to JPG conversion...")
    
    raw_dir = Path(params['paths']['raw_data'])
    emotions = params['data']['classes']
    
    total_converted = 0
    total_failed = 0
    total_skipped = 0
    
    # Process each emotion folder
    for emotion_id, emotion_name in emotions.items():
        emotion_dir = raw_dir / emotion_name
        
        if not emotion_dir.exists():
            print(f"⚠️  Folder not found: {emotion_dir}, skipping...")
            continue
        
        print(f"\n📂 Processing: {emotion_name}")
        
        # Find all DNG files
        dng_files = list(emotion_dir.glob('*.dng')) + \
                    list(emotion_dir.glob('*.DNG'))
        
        if not dng_files:
            print(f"   No DNG files found")
            continue
        
        print(f"   Found {len(dng_files)} DNG files")
        
        # Convert each DNG file
        for dng_file in tqdm(dng_files, desc=f"   Converting {emotion_name}"):
            # Create JPG filename (same name, different extension)
            jpg_file = dng_file.with_suffix('.jpg')
            
            # Skip if JPG already exists
            if jpg_file.exists():
                total_skipped += 1
                continue
            
            # Convert DNG to JPG
            success = convert_dng_to_jpg(dng_file, jpg_file, quality=95)
            
            if success:
                total_converted += 1
                # Optionally delete DNG file after conversion (uncomment if you want)
                # dng_file.unlink()
            else:
                total_failed += 1
    
    # Summary
    print(f"\n✅ Conversion complete!")
    print(f"📊 Summary:")
    print(f"   ✅ Converted: {total_converted} files")
    print(f"   ⏭️  Skipped (already exists): {total_skipped} files")
    print(f"   ❌ Failed: {total_failed} files")
    print(f"   📁 Total JPG files now available: {total_converted + total_skipped}")
    
    return total_converted, total_skipped, total_failed

def verify_conversion(params):
    """
    Verify that all DNG files have been converted
    """
    print("\n🔍 Verifying conversion...")
    
    raw_dir = Path(params['paths']['raw_data'])
    emotions = params['data']['classes']
    
    for emotion_id, emotion_name in emotions.items():
        emotion_dir = raw_dir / emotion_name
        
        if not emotion_dir.exists():
            continue
        
        # Count files
        dng_count = len(list(emotion_dir.glob('*.dng'))) + \
                    len(list(emotion_dir.glob('*.DNG')))
        jpg_count = len(list(emotion_dir.glob('*.jpg'))) + \
                    len(list(emotion_dir.glob('*.JPG')))
        png_count = len(list(emotion_dir.glob('*.png'))) + \
                    len(list(emotion_dir.glob('*.PNG')))
        
        total_images = jpg_count + png_count
        
        print(f"\n📂 {emotion_name:12s}:")
        print(f"   DNG files: {dng_count}")
        print(f"   JPG files: {jpg_count}")
        print(f"   PNG files: {png_count}")
        print(f"   Total usable images: {total_images}")
        
        if dng_count > 0 and jpg_count == 0:
            print(f"   ⚠️  Warning: DNG files not converted!")

if __name__ == "__main__":
    # Load parameters
    params = load_params()
    
    # Convert all DNG files
    converted, skipped, failed = convert_all_dng_files(params)
    
    # Verify conversion
    verify_conversion(params)
    
    print("\n✅ Ready to run preprocessing!")