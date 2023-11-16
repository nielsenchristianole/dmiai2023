
# import PATH
from pathlib import Path
import os
from PIL import Image

import torchvision
from augmentor import perspective_transform, to_grayscale, project_to_cam, extract_corners
import torch

from torchvision.transforms.functional import InterpolationMode
import numpy as np
from models.unet_bigger import UNet
import matplotlib.pyplot as plt
from utils import dice_score as dice
import tqdm
import cv2


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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



idx = [1,13,20,21,34,43,48,49,50,52,53,54,58,71,80,87,88,91,
  99,102,105,106,121,130,134,149,151,160,161,166,169,174,187,188,189,190
,191,201,205,207,212,214,217,235,236,241,243,251,252,257,259,263,264,269
,270,273,276,293,295,303,306,308,309,313,315,319,328,330,339,343,344,345
,348,350,359,363,366,372,385,395,396,415,417,422,425,429,431]

fig, axs = plt.subplots(1,5, figsize = (25,5))

dices = []

mask_average = np.zeros((512,512))
for i, (img_path, mask_path) in enumerate(zip(Path('data/all_images').glob('*.png'), Path('data/all_masks').glob('*.png'))):

    if i in idx:
        continue

for i, (img_path, mask_path) in enumerate(zip(Path('data/all_images').glob('*.png'), Path('data/all_masks').glob('*.png'))):

    if i not in idx:
        continue

    img = np.array(Image.open(img_path))[:,:,:3]
    label_true = np.array(Image.open(mask_path))[:,:,:3]

    img_gray = to_grayscale(img)[None,None,:]

    pred = model(torch.tensor(img_gray, dtype=torch.float32, device = DEVICE))

    pred = pred[0][0].cpu().detach().numpy() > 0.5

    # # Perform dialation on the prediction
    # pred = pred*255
    # pred = pred.astype(np.uint8)
    # kernel = np.ones((3,3),np.uint8)
    # pred = cv2.dilate(pred,kernel,iterations = 1)
    # pred = pred.astype(np.float32)/255

    dices.append(dice(label_true, np.stack((pred,pred,pred)).transpose((1,2,0))))

    # fig, axs = plt.subplots(1,5, figsize = (25,5))

    # axs[0].imshow(img, cmap = 'gray')
    # axs[1].imshow(label_true, cmap = 'gray')
    # axs[2].imshow(pred, cmap = 'gray')
    # plt.show()

    # print(dice(label_true, ))

print(np.mean(dices))

# labels, label_pred = ensemble_perspective_prediction(img, model, 10)







# 




plt.show()
