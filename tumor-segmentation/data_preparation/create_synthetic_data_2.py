import os
import glob
import copy
from collections import defaultdict

import cv2
import tqdm
import numpy as np
from PIL import Image, ImageOps
from PIL.Image import Resampling
import matplotlib.pyplot as plt

CONTROL_IMG_DIR = './data/controls/imgs'
PATIENT_IMG_DIR = './data/patients/imgs'
PATIENT_MASK_DIR = './data/patients/labels'

SYNTHETIC_IMG_DIR = './data/training_data'
os.makedirs(SYNTHETIC_IMG_DIR, exist_ok=True)
os.makedirs(os.path.join(SYNTHETIC_IMG_DIR, 'imgs'), exist_ok=True)
os.makedirs(os.path.join(SYNTHETIC_IMG_DIR, 'labels'), exist_ok=True)


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

def read_path(dir: str, padding: tuple[int, int, int], is_label: bool) -> dict[int, np.ndarray]:
    data_dict = dict()
    for img_path in glob.glob(os.path.join(dir, '*.png')):
        img = cv2.imread(img_path)
        img = add_padding_bottom_only(Image.fromarray(img), target_size, padding_color=padding, is_label=is_label)
        
        key = get_num(img_path)
        data_dict[key] = np.array(img)
    return data_dict

control_imgs = read_path(CONTROL_IMG_DIR, (255, 255, 255), False)
patient_imgs = read_path(PATIENT_IMG_DIR, (255, 255, 255), False)
patient_masks = read_path(PATIENT_MASK_DIR, (0, 0, 0), True)

threshold = 245
patient_contours = {k: cv2.findContours((v[:, :, 0] < threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for k, v in patient_imgs.items()}
control_contours = {k: cv2.findContours((v[:, :, 0] < threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for k, v in control_imgs.items()}


patient_person_mask = dict()
for k, countour in patient_contours.items():
    filled_contour = cv2.drawContours(np.zeros_like(patient_imgs[k]), countour, -1, (255, 255, 255), thickness=cv2.FILLED)[:, :, 0]
    patient_person_mask[k] = filled_contour

control_person_mask = dict()
for k, countour in control_contours.items():
    filled_contour = cv2.drawContours(np.zeros_like(control_imgs[k]), countour, -1, (255, 255, 255), thickness=cv2.FILLED)[:, :, 0]
    control_person_mask[k] = filled_contour

cancer_cloned = defaultdict(int)
max_cancer_cloned = 5

def get_nearest_neighbor(control_k) -> int:
    max_iou = 0
    min_img_key = None
    c_p_mask = control_person_mask[control_k]
    for k, v in patient_person_mask.items():
        if cancer_cloned[k] >= max_cancer_cloned:
            continue
        iou = np.sum(c_p_mask & v) / np.sum(c_p_mask | v)
        if max_iou < iou:
            max_iou = iou
            min_img_key = k
    cancer_cloned[min_img_key] += 1
    return min_img_key


for control_key, control_img in tqdm.tqdm(control_imgs.items(), disable=False):
    patient_key = get_nearest_neighbor(control_key)
    patient_img = patient_imgs[patient_key]
    patient_mask = patient_masks[patient_key]

    c_p_mask = control_person_mask[control_key]

    synthetic_img = copy.deepcopy(control_img)
    synthetic_mask = patient_mask & np.stack([c_p_mask, c_p_mask, c_p_mask], axis=-1)

    cancer_idxs = np.argwhere(synthetic_mask)

    if len(cancer_idxs) == 0:
        continue

    for idx in cancer_idxs:
        synthetic_img[*idx] = patient_img[*idx]

    Image.fromarray(synthetic_img.astype(np.uint8)).save(os.path.join(SYNTHETIC_IMG_DIR, f'imgs/synthetic_patient_{control_key:0>3}.png'))
    Image.fromarray(synthetic_mask.astype(np.uint8)).save(os.path.join(SYNTHETIC_IMG_DIR, f'labels/synthetic_segmentation_{control_key:0>3}.png'))
    