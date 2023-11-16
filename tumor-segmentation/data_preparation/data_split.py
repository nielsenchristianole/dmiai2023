#!/usr/bin/env python3
import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(patient_folder, mask_folder, destination, train_size, val_size):
    assert os.path.exists(patient_folder), f"Folder {patient_folder} does not exist."
    assert os.path.exists(mask_folder), f"Folder {mask_folder} does not exist."

    # Get the list of patient files and mask files
    patient_files = os.listdir(patient_folder)
    mask_files = set(os.listdir(mask_folder))

    # Filter out patient files that have a corresponding mask file
    patient_files_with_mask = [f for f in patient_files if f in mask_files]

    # Split the data into train, validation, and test sets
    train_files_with_mask, temp_files_with_mask = train_test_split(patient_files_with_mask, train_size=train_size, random_state=42)
    val_size_adjusted = val_size / (1 - train_size)
    val_files_with_mask, test_files_with_mask = train_test_split(temp_files_with_mask, train_size=val_size_adjusted, random_state=42)

    # Function to copy patient and corresponding mask files
    def copy_files(file_list, source_patient_folder, source_mask_folder, dest_patient_folder, dest_mask_folder):
        for file in file_list:
            shutil.copy(os.path.join(source_patient_folder, file), os.path.join(dest_patient_folder, file))
            if file in mask_files:
                shutil.copy(os.path.join(source_mask_folder, file), os.path.join(dest_mask_folder, file))

    # Copy files to train, validation, and test folders
    for dtype, file_list in zip(['train', 'val', 'test'], [train_files_with_mask, val_files_with_mask, test_files_with_mask]):
        patient_dest = os.path.join(destination, dtype, 'patient')
        mask_dest = os.path.join(destination, dtype, 'mask')
        os.makedirs(patient_dest, exist_ok=True)
        os.makedirs(mask_dest, exist_ok=True)
        copy_files(file_list, patient_folder, mask_folder, patient_dest, mask_dest)

    # Print the number of files in each dataset
    for dtype in ['train', 'val', 'test']:
        patient_dest = os.path.join(destination, dtype, 'patient')
        mask_dest = os.path.join(destination, dtype, 'mask')
        num_patient_files = len(os.listdir(patient_dest))
        num_mask_files = len(os.listdir(mask_dest)) if os.path.exists(mask_dest) else 0
        print(f"{dtype.upper()}: {num_patient_files} patient files, {num_mask_files} mask files")

    print("Datasets created.")

split_data('data/all_images', 'data/all_masks', 'data_training', train_size=0.85, val_size=0.15)
