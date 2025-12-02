import os
import csv
from pathlib import Path

def create_img_path_csv(folder, output_csv):
    row_data = ['Image', 'Mask']
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row_data)

        subfolders = ['benign', 'malignant']
        for subfolder in subfolders:
            for i in range(500):
                image_path = Path(folder) / subfolder / f"{subfolder} ({i}).png"
                mask_path = Path(folder) / subfolder / f"{subfolder} ({i})_mask.png"
                if image_path.exists() and mask_path.exists():
                    writer.writerow([image_path, mask_path])

image_folder_path = '/Users/abby/.cache/kagglehub/datasets/aryashah2k/breast-ultrasound-images-dataset/versions/1/Dataset_BUSI_with_GT'
csv_dest_path = '/Users/abby/Desktop/image_mask_path.csv'
create_img_path_csv(image_folder_path, csv_dest_path)