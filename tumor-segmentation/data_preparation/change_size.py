#!/usr/bin/env python3
import os
from PIL import Image

# Folders for original images and labels
img_folder = 'data/patients/imgs'
label_folder = 'data/patients/labels'

# Folders for resized images and labels
resized_img_folder = 'data/resized_imgs'
resized_label_folder = 'data/resized_labels'

# Create the directories if they don't exist
os.makedirs(resized_img_folder, exist_ok=True)
os.makedirs(resized_label_folder, exist_ok=True)

# Interpolation method for resizing
interpolation_method = Image.NEAREST

# Resize patient images
for img_file in os.listdir(img_folder):
    if img_file.endswith('.png') and img_file.startswith('patient_'):
        original_img_path = os.path.join(img_folder, img_file)
        resized_img_path = os.path.join(resized_img_folder, img_file)
        img = Image.open(original_img_path)
        img_resized = img.resize((512, 512), interpolation_method)
        img_resized.save(resized_img_path)

# Resize label images
for img_file in os.listdir(label_folder):
    if img_file.endswith('.png') and img_file.startswith('segmentation_'):
        original_img_path = os.path.join(label_folder, img_file)
        resized_img_path = os.path.join(resized_label_folder, img_file)
        img = Image.open(original_img_path)
        img_resized = img.resize((512, 512), interpolation_method)
        img_resized.save(resized_img_path)
