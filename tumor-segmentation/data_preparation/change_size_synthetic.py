import os
from PIL import Image, ImageOps

synthetic_images_folder = 'data/synthetic_images_plotted'
synthetic_mask_folder = 'data/synthetic_mask'

# New folders to save the padded images and labels
padded_img_folder = 'data/resized_synthetic_imgs'
padded_label_folder = 'data/resized_synthetic_masks'

# Create the directories if they don't exist
os.makedirs(padded_img_folder, exist_ok=True)
os.makedirs(padded_label_folder, exist_ok=True)

target_size = (400, 1024)

def add_padding_bottom_only(img, target_size, padding_color):
    """
    Adds padding to the bottom of the image to reach the target size.
    Padding color is specified by padding_color.
    """
    original_size = img.size
    delta_width = target_size[0] - original_size[0]
    delta_height = target_size[1] - original_size[1]
    padding = (0, 0, delta_width, delta_height)  # Padding only at the bottom
    return ImageOps.expand(img, padding, fill=padding_color)

for img_file in os.listdir(synthetic_images_folder):
    if img_file.endswith('.png') and img_file.startswith('synthetic_'):
        # Paths to the original and padded image
        original_img_path = os.path.join(synthetic_images_folder, img_file)
        padded_img_path = os.path.join(padded_img_folder, img_file)
        img = Image.open(original_img_path)
        img_padded = add_padding_bottom_only(img, target_size, padding_color=(255, 255, 255))
        img_padded.save(padded_img_path)

for img_file in os.listdir(synthetic_mask_folder):
    if img_file.endswith('.png') and img_file.startswith('synthetic_'):
        original_img_path = os.path.join(synthetic_mask_folder, img_file)
        padded_img_path = os.path.join(padded_label_folder, img_file)
        img = Image.open(original_img_path)
        img_padded = add_padding_bottom_only(img, target_size, padding_color=(0, 0, 0))
        img_padded.save(padded_img_path)
