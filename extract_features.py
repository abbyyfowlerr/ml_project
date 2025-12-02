import pandas as pd
import radiomics 
from radiomics import featureextractor 
import logging
import os
import sys
import SimpleITK as sitk 

# --- Configuration ---
INPUT_CSV = '/Users/abby/Desktop/image_mask_path.csv'
OUTPUT_CSV = '/Users/abby/Desktop/radiomics_features_extracted.csv'
TARGET_LABEL = 255  # Correct label for the 'white' object

# --- Define Image Processing Settings ---
settings_dict = {
    'label': TARGET_LABEL,
    'featureClass': {
        'shape2D': None # Enable shape2D for 2D inputs
    },
    'imageProcessing': {
        'correctMask': True, 
        'force2D': True
    }
}

# --- Setup Logging ---
try:
    radiomics.setup_logging(logFilepath=None, level=logging.INFO)
except AttributeError:
    print("Warning: Could not set up radiomics logging.", file=sys.stderr)
    pass 
logger = logging.getLogger('pyradiomics')


def pre_load_image(filepath, is_mask=False):
    """
    Forces the loading of an image/mask as a single-channel, 8-bit grayscale image (Fixes Color Error),
    and casts it to the correct numerical type (Fixes Indexing Error).
    """
    # 1. Force read as single-channel 8-bit image to fix the 'vector of 8-bit' color error
    image = sitk.ReadImage(filepath, sitk.sitkUInt8)
    
    if is_mask:
        # 2. CRITICAL FIX: Cast mask to a robust integer type (Label Maps must be integers)
        image = sitk.Cast(image, sitk.sitkInt32)
    else:
        # 3. Fix: Cast image data to float for feature calculation precision
        image = sitk.Cast(image, sitk.sitkFloat64) 
        
    return image


def extract_features_from_csv(input_csv, output_csv, settings):
    """
    Reads a CSV, extracts radiomics features for each image/mask pair,
    and saves the results to a new CSV.
    """
    if not os.path.exists(input_csv):
        logger.error(f"Input file not found: {input_csv}")
        return

    logger.info(f"Loading data from {input_csv}...")
    
    # 1. Load Data
    try:
        data_df = pd.read_csv(input_csv)
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return

    # 2. Initialize Extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    logger.info(f"Extractor initialized. Target label: {settings['label']}. Total files to process: {len(data_df)}")
    
    all_features = []
    
    # 3. Iterate and Extract
    for index, row in data_df.iterrows():
        image_filepath = row['Image']
        mask_filepath = row['Mask']
        
        logger.info(f"Processing file {index + 1}/{len(data_df)}: {os.path.basename(image_filepath)}")
        
        if not os.path.exists(image_filepath) or not os.path.exists(mask_filepath):
            logger.warning(f"Skipping row {index + 1}: Image or Mask file not found at path.")
            continue

        try:
            # Load the images and masks using the custom function with explicit casting
            sitk_image = pre_load_image(image_filepath, is_mask=False)
            sitk_mask = pre_load_image(mask_filepath, is_mask=True) 

            # Execute feature extraction using the SimpleITK objects
            feature_vector = extractor.execute(
                imageFilepath=sitk_image, 
                maskFilepath=sitk_mask
            )
            
            # Prepare results dictionary
            result = dict(feature_vector)
            result['Original_Image_Path'] = image_filepath
            result['Original_Mask_Path'] = mask_filepath
            
            all_features.append(result)
            
            logger.info(f"Successfully extracted features for index {index + 1}")

        except Exception as e:
            logger.error(f"Failed to extract features for {image_filepath}. Error: {e}")
            # Continue to the next file if one fails
            
    # 4. Save Results
    if all_features:
        features_df = pd.DataFrame(all_features)
        features_df.to_csv(output_csv, index=False)
        logger.info(f"\nFeature extraction complete. Results saved to {output_csv}")
    else:
        logger.warning("No features were successfully extracted.")

if __name__ == '__main__':
    extract_features_from_csv(INPUT_CSV, OUTPUT_CSV, settings_dict)