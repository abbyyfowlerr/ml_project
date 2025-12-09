import os
from PIL import Image

# 1. Source: Where the original raw data is
SOURCE_ROOT = '/Users/abby/.cache/kagglehub/datasets/aryashah2k/breast-ultrasound-images-dataset/versions/1/Dataset_BUSI_with_GT'

# 2. Destination: Where your resized images currently sit
OUTPUT_ROOT = '/Users/abby/cisc684/ml_project/.data/processed_busi'

# 3. Settings
CLASSES = ['benign', 'malignant']
TARGET_SIZE = (224, 224) 

def process_masks():
    print("--- Starting Mask Processing for BUSI ---")
    for category in CLASSES:
        source_dir = os.path.join(SOURCE_ROOT, category)
        dest_dir = os.path.join(OUTPUT_ROOT, f"{category}_sub")
        
        # Safety check
        if not os.path.exists(dest_dir):
            print(f"❌ Error: Destination folder not found: {dest_dir}")
            print("   (Did you delete the folders? You need to run the image script first.)")
            continue

        print(f"Processing masks for: {category}...")
        
        count = 0
        for filename in os.listdir(source_dir):
            # We ONLY want files with '_mask' this time
            if '_mask' in filename and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(source_dir, filename)
                dst_path = os.path.join(dest_dir, filename)

                try:
                    with Image.open(src_path) as img:
                        # IMPORTANT: Use NEAREST to keep masks black/white (no blurry edges)
                        img_resized = img.resize(TARGET_SIZE, Image.Resampling.NEAREST)
                        img_resized.save(dst_path)
                        count += 1
                except Exception as e:
                    print(f"   Failed to process {filename}: {e}")
        
        print(f"✅ Finished {category}: {count} masks saved.")

if __name__ == "__main__":
    process_masks()