import os
import glob
import copy

import cv2
import tqdm
import numpy as np
from PIL import Image, ImageOps
from PIL.Image import Resampling


CONTROL_IMG_DIR = './data/controls/imgs'
PATIENT_IMG_DIR = './data/patients/imgs'
PATIENT_MASK_DIR = './data/patients/labels'

SYNTHETIC_IMG_DIR = './data/synthetic'
os.makedirs(SYNTHETIC_IMG_DIR, exist_ok=True)
os.makedirs(os.path.join(SYNTHETIC_IMG_DIR, 'imgs'), exist_ok=True)
os.makedirs(os.path.join(SYNTHETIC_IMG_DIR, 'labels'), exist_ok=True)

control_imgs = dict()
patient_imgs = dict()
patient_masks = dict()


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

get_num = lambda x: int(x.split('_')[-1].split('.')[0])

for img_path in glob.glob(os.path.join(CONTROL_IMG_DIR, '*.png')):
    img = cv2.imread(img_path)
    img = add_padding_bottom_only(Image.fromarray(img), target_size, padding_color=(255, 255, 255), is_label=False)
    
    key = get_num(img_path)
    control_imgs[key] = np.array(img)

for img_path in glob.glob(os.path.join(PATIENT_IMG_DIR, '*.png')):
    img = cv2.imread(img_path)
    img = add_padding_bottom_only(Image.fromarray(img), target_size, padding_color=(255, 255, 255), is_label=False)
    
    key = get_num(img_path)
    patient_imgs[key] = np.array(img)

for img_path in glob.glob(os.path.join(PATIENT_MASK_DIR, '*.png')):
    img = cv2.imread(img_path)
    img = add_padding_bottom_only(Image.fromarray(img), target_size, padding_color=(0, 0, 0), is_label=True)
    
    key = get_num(img_path)
    patient_masks[key] = np.array(img)


def get_nearest_neighbor(img: np.ndarray, imgs: dict[int, np.ndarray]) -> int:
    min_dist = float('inf')
    min_img_key = None
    for k, v in imgs.items():
        dist = np.linalg.norm(img - v)
        if dist < min_dist:
            min_dist = dist
            min_img_key = k
    return min_img_key


for control_key, control_img in tqdm.tqdm(control_imgs.items(), disable=False):
    patient_key = get_nearest_neighbor(control_img, patient_imgs)
    patient_img = patient_imgs[patient_key]
    patient_mask = patient_masks[patient_key]

    synthetic_img = copy.deepcopy(control_img)
    synthetic_mask = copy.deepcopy(patient_mask)
    
    cancer_idxs = np.argwhere(patient_mask)
    # print(cancer_idxs, '\n', cancer_idxs.shape)
    # break
    for idx in cancer_idxs:
        # print(idx, synthetic_img[idx[:2]])
        synthetic_img[*idx] = patient_img[*idx]

    Image.fromarray(synthetic_img.astype(np.uint8)).save(os.path.join(SYNTHETIC_IMG_DIR, f'imgs/patient_{control_key:0>3}.png'))
    Image.fromarray(synthetic_mask.astype(np.uint8)).save(os.path.join(SYNTHETIC_IMG_DIR, f'labels/segmentation_{control_key:0>3}.png'))
    