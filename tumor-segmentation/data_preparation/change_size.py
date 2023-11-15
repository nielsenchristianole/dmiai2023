#!/bin/bash
import os
from PIL import Image

img_folder = './tumor-segmentation/data/patients/imgs'
label_folder = './DM-i-AI-2023/tumor-segmentation/data/patients/labels'

# New folders to save the resized images and labels
resized_img_folder = './tumor-segmentation/data/resized_imgs'
resized_label_folder = './tumor-segmentation/data/resized_labels'

interpolation_method = Image.NEAREST

for img_file in os.listdir(img_folder):
    if img_file.endswith('.png') and img_file.startswith('patient_'):
        # Paths to the original and resized image
        original_img_path = os.path.join(img_folder, img_file)
        resized_img_path = os.path.join(resized_img_folder, img_file)
        img = Image.open(original_img_path)
        img_resized = img.resize((512, 512), interpolation_method)
        img_resized.save(resized_img_path)

for img_file in os.listdir(label_folder):
    if img_file.endswith('.png') and img_file.startswith('segmentation_'):
        original_img_path = os.path.join(label_folder, img_file)
        resized_img_path = os.path.join(resized_label_folder, img_file)
        img = Image.open(original_img_path)
        img_resized = img.resize((512, 512), interpolation_method)
        img_resized.save(resized_img_path)