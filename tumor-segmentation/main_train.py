import random

import cv2
import numpy as np
from pathlib import Path

import torch
import augmentor

# import augmentor

def create_color_gradient_tensor(width, height):
    
    # Create a linear gradient from 0 to 1 with length equal to the height of the gradient
    r = np.linspace(0, 1, width)
    g = np.linspace(0, 1, height)
    
    # Create a meshgrid from the linear gradients
    r, g = np.meshgrid(r, g)
    b = np.flip((r + g)/2)
    
    # Stack the color channels together
    colors = np.stack([r, g, b], axis = 2)
    
    coordinates = np.stack(np.meshgrid(np.arange(0, width, 1),
                                          np.arange(0, height, 1)), axis=1)
    
    coordinates = coordinates.transpose((2,0,1))
    
    # Add a z-coordinate of value 0 to the coordinates
    z = np.zeros((width, height, 1))
    coordinates = np.concatenate([coordinates, z], axis=2)
    
    # Colors: RGB value for all pixels (height, width, 3)
    # Locations of pixels in 3D space (height, width, 3)
    return colors, coordinates


img = cv2.imread("data/patients/imgs/patient_001.png")
label = cv2.imread("data/patients/labels/segmentation_001.png")

preprossesor = augmentor.ImagePreProcessor(image_size=512)
aug = augmentor.Augmentor()

img, label = preprossesor([img], [label])
img, label = aug(img, label)

import matplotlib.pyplot as plt

plt.imshow(img[0])
plt.show()
plt.imshow(label[0])
plt.show()




            







# img1, _ = create_color_gradient_tensor(100, 100)
# img2, _ = create_color_gradient_tensor(600, 100)
# img3, _ = create_color_gradient_tensor(100, 600)
# img4, _ = create_color_gradient_tensor(800, 700)
# img5, _ = create_color_gradient_tensor(800, 800)

# images = [img1,img2,img3,img4,img5]
# labels = [img1,img2,img3,img4,img5]

# processor = ImagePreProcessor(image_size=512)

# new_images, new_labels = processor(images, labels)

# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(5, 2)

# for i in range(5):
#     axs[i, 0].imshow(new_images[i])
#     axs[i, 1].imshow(new_labels[i])

# plt.show()
