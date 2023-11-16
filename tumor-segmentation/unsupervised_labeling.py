
# import PATH
from pathlib import Path
import os
from PIL import Image

import torchvision
from augmentor import perspective_transform, to_grayscale, project_to_cam, extract_corners
import torch

from torchvision.transforms.functional import InterpolationMode
import numpy as np
from models.unet import UNet
import matplotlib.pyplot as plt
from utils import dice_score as dice
import tqdm


DEVICE = 'cpu'

def ensemble_perspective_prediction(img, model, num_perspective):

    labels = []

    for i in tqdm.trange(num_perspective):
        corners = extract_corners(img)
        corners = project_to_cam(corners).T
        perspective, _, corners_new = perspective_transform(img)

        perspective = to_grayscale(perspective)[None,None,:]

        label_perspective = model(torch.tensor(perspective, dtype=torch.float32, device = DEVICE))
        
        label = torchvision.transforms.functional.perspective(label_perspective[0],
                                                                corners_new.tolist(),
                                                                corners.tolist(),
                                                                fill = 0,
                                                                interpolation = InterpolationMode.NEAREST).permute(1,2,0).cpu().numpy()
        label = label > 0.5
        
        labels.append(label)

    label = np.stack(labels, axis = 0).mean(axis = 0)

    return labels, label


model = UNet(device = DEVICE).requires_grad_(False)


model.load_state_dict(torch.load('unet_pet_segmentation_1658_best_1.pth'))

img = np.array(Image.open('data/all_images/patient_001.png'))[:,:,:3]
label_true = np.array(Image.open('data/all_masks/patient_001.png'))[:,:,:3]


labels, label_pred = ensemble_perspective_prediction(img, model, 10)


fig, axs = plt.subplots(1,5, figsize = (25,5))

axs[0].imshow(img, cmap = 'gray')
axs[1].imshow(label_true, cmap = 'gray')
axs[2].imshow(label_pred, cmap = 'gray')
axs[3].imshow(label_pred > 0.5, cmap = 'gray')
axs[4].imshow(labels[0], cmap = 'gray')

print(dice(label_true, label_pred > 0.5))


plt.show()
