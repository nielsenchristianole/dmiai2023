import os
from PIL import Image, ImageOps
from PIL.Image import Resampling

# Folders for original images and labels
img_folder = 'data/patients/imgs'
label_folder = 'data/patients/labels'

# Folders for padded images and labels
padded_img_folder = 'data/training_data/imgs'
padded_label_folder = 'data/training_data/labels'

# Create the directories if they don't exist
os.makedirs(padded_img_folder, exist_ok=True)
os.makedirs(padded_label_folder, exist_ok=True)

target_size = (400, 1024)

def add_padding_bottom_only(img, target_size, padding_color, is_label):
    """
    Adds padding to the bottom of the image to reach the target size.
    Padding color is specified by padding_color.
    """
    original_size = img.size
    delta_width = target_size[0] - original_size[0]
    # delta_height = target_size[1] - original_size[1]
    
    top_padding = 1024 - 991
    bottom_padding = 1024 - img.size[1] - top_padding
    
    padding = (delta_width//2, top_padding, delta_width//2, bottom_padding)  # Padding only at the bottom

    img = ImageOps.expand(img, padding, fill=padding_color)

    assert img.size == (400, 1024), "image is scaled incorrectly"

    interpolation = Resampling.NEAREST if is_label else Resampling.LANCZOS

    return img.resize((200, 512), resample = interpolation)

# Add padding to images
for img_file in os.listdir(img_folder):
    if img_file.endswith('.png') and img_file.startswith('patient_'):
        original_img_path = os.path.join(img_folder, img_file)
        padded_img_path = os.path.join(padded_img_folder, img_file)
        img = Image.open(original_img_path)
        img_padded = add_padding_bottom_only(img, target_size, padding_color=(255, 255, 255), is_label = False)
        img_padded.save(padded_img_path)

# Add padding to labels
for img_file in os.listdir(label_folder):
    if img_file.endswith('.png') and img_file.startswith('segmentation_'):
        original_img_path = os.path.join(label_folder, img_file)
        padded_img_path = os.path.join(padded_label_folder, img_file)
        img = Image.open(original_img_path)
        img_padded = add_padding_bottom_only(img, target_size, padding_color=(0, 0, 0), is_label = True)
        img_padded.save(padded_img_path)
