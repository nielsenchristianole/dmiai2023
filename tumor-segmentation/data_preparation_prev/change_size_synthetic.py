import os
from PIL import Image

synthetic_images_folder = 'data/synthetic_images_plotted'
synthetic_mask_folder = 'data/synthetic_mask'

# New folders to save the resized images and labels
resized_img_folder = 'data/resized_synthetic_imgs'
resized_label_folder = 'data/resized_synthetic_masks'

# Create the directories if they don't exist
os.makedirs(resized_img_folder, exist_ok=True)
os.makedirs(resized_label_folder, exist_ok=True)


interpolation_method = Image.NEAREST

for img_file in os.listdir(synthetic_images_folder):
    if img_file.endswith('.png') and img_file.startswith('synthetic_'):
        # Paths to the original and resized image
        original_img_path = os.path.join(synthetic_images_folder, img_file)
        resized_img_path = os.path.join(resized_img_folder, img_file)
        img = Image.open(original_img_path)
        img_resized = img.resize((512, 512), interpolation_method)
        img_resized.save(resized_img_path)

for img_file in os.listdir(synthetic_mask_folder):
    if img_file.endswith('.png') and img_file.startswith('synthetic_'):
        original_img_path = os.path.join(synthetic_mask_folder, img_file)
        resized_img_path = os.path.join(resized_label_folder, img_file)
        img = Image.open(original_img_path)
        img_resized = img.resize((512, 512), interpolation_method)
        img_resized.save(resized_img_path)