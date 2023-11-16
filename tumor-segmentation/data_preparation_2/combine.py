import os
import shutil

def merge_folders(source_folders, destination_folder):
    """
    Merge contents of multiple source folders into a single destination folder.
    """
    for folder in source_folders:
        for file_name in os.listdir(folder):
            source_file = os.path.join(folder, file_name)
            destination_file = os.path.join(destination_folder, file_name)
            shutil.copy(source_file, destination_file)

def rename_files_in_folder(folder, old_str, new_str):
    """
    Rename files in a folder by replacing a specified substring in filenames.
    """
    for file_name in os.listdir(folder):
        if old_str in file_name:
            new_file_name = file_name.replace(old_str, new_str)
            os.rename(os.path.join(folder, file_name), os.path.join(folder, new_file_name))

# Define your folder paths here
image_folders = ['data/padded_imgs', 'data/padded_synthetic_imgs']
mask_folders = ['data/padded_labels', 'data/padded_synthetic_masks']
merged_image_folder = 'data/all_images'
merged_mask_folder = 'data/all_masks'

os.makedirs(merged_image_folder, exist_ok=True)
os.makedirs(merged_mask_folder, exist_ok=True)

# Merge image folders
merge_folders(image_folders, merged_image_folder)

# Merge mask folders and rename files
merge_folders(mask_folders, merged_mask_folder)
rename_files_in_folder(merged_mask_folder, 'segmentation_', 'patient_')
