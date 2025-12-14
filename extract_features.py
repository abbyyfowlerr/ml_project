import SimpleITK as sitk
from PIL import Image
import numpy as np
import os
import csv
from radiomics import featureextractor

# path to your CSV file
CSV_FILE = "image_mask_path.csv"
# name for the output file
OUTPUT_FILE = "radiomics_features_with_filters.csv"

def load_and_synchronize_image(image_path, mask_path):
    """
    Loads, synchronizes, and cleans the image and mask pair, 
    applying the fixes needed for pyradiomics stability.
    Returns (sitk_image, sitk_mask) or (None, None) on failure.
    """
    try:
        # process image
        image_pil = Image.open(image_path)
        np_img = np.array(image_pil.convert("L"))
        image = sitk.GetImageFromArray(np_img)

        # process mask
        mask = sitk.ReadImage(mask_path)
        mask = sitk.Cast(mask, sitk.sitkUInt8)
        
        return image, mask
    except FileNotFoundError:
        print(f"File not found. Skipping image: {os.path.basename(image_path)}")
        return None, None


def run_batch_extraction(param_file):
    extractor = featureextractor.RadiomicsFeatureExtractor(param_file)

    all_results = []
    fieldnames = ["ID"]

    try:
        with open(CSV_FILE, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            next(reader) # Skip header

            for row in reader:
            #for row in reader:
                image_path = row[0].strip()
                mask_path = row[1].strip()

                # --- Use Filename as ID ---
                # Example: "/path/to/benign (1).png" -> "benign (1)"
                # Remove path and extension
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                
                image, mask = load_and_synchronize_image(image_path, mask_path)

                if image is None:
                    # Log minimal error to console for failed files
                    print(f"Skipping failed/missing file: {base_filename}")
                    continue
                
                # --- Feature Extraction ---
                try:
                    feature_vector = extractor.execute(image, mask)

                    row_data = {"ID": base_filename}
                    
                    for key, value in feature_vector.items():
                        # Only include non-general info keys
                        if not key.startswith('general_'):
                            # Clean up keys for CSV
                            row_data[key] = float(value) if isinstance(value, (np.float64, np.float32)) else value

                    all_results.append(row_data)

                    # Update fieldnames from the first successful row
                    if len(fieldnames) == 1:
                        fieldnames.extend(list(row_data.keys())[1:])

                except Exception as e:
                    print(f"Extraction failed for {base_filename}: {e}")
                    continue

    except FileNotFoundError:
        print(f"\nERROR: Input file not found at {CSV_FILE}.")
        return

    # Write results to CSV 
    if all_results:
        try:
            with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
            print(f"\nSUCCESS: Results saved to {OUTPUT_FILE} for {len(all_results)} images.")
        except Exception as e:
            print(f"ERROR: Could not write results to CSV. {e}")
    else:
        print("\nCOMPLETED: No features were successfully extracted.")

if __name__ == "__main__":
    run_batch_extraction("settings.yml")