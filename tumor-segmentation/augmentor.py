#%%
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import random
import cv2

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

def extract_corners(image : np.ndarray) -> np.ndarray:
    """
    Extracts the four corners of an image tensor and returns them as a tensor.
    Either a single image or a batch of images can be passed to this function.
    
    Args:
    - image: A (b,w,h,3)-dimensional torch.Tensor representing an image.
    
    Returns:
    - corners: A (b,4,3)-dimensional torch.Tensor representing the four corners of the input image.
    """
    
    corners = np.array([[0,0,0],
                            [0,image.shape[1],0],
                            [image.shape[0],0,0],
                            [image.shape[0],image.shape[1],0]])
    
    return corners

def project_to_cam(coord : np.ndarray) -> np.ndarray:

    coord = np.concatenate((coord, np.ones(coord.shape[:-1] + (1,))), axis=-1)
    
    # 3x4 camera matrix
    cam_matrix = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0]])
    # 3x3 camera intrinsics matrix
    cam_intrinsics = np.array([[1, 0, 1],
                               [0, 1, 1],
                               [0, 0, 1]])
    
    proj_coord = cam_intrinsics@cam_matrix@coord.T
    
    return proj_coord[:2,:]
    

class Homography:
    
    def __init__(self, rot_xyz = np.array([0,0,0]),
                       shear = np.array([0,0]),
                       scale = np.array([0,0]),
                       shift = np.array([0,0])):
        
        self.transform_matrix = self._get_rot_matrix(rot_xyz)@ \
                                self._get_shear_matrix(shear)@ \
                                self._get_scale_matrix(scale)
        
        # self.transform_matrix = self._get_shear_matrix(shear)
        # self.transform_matrix = self._get_scale_matrix(scale)
        
        self.shift = np.concatenate((shift, np.zeros(1)))
        self.rot_xyz = rot_xyz
        self.shear = shear
        self.scale = scale
    
    def __call__(self, coord) -> np.ndarray:
        c_range_half = (coord.max() - coord.min())/2
        
        return (coord - c_range_half)@self.transform_matrix + self.shift*c_range_half + c_range_half

    def _get_rot_matrix(self, rot_xyz : np.ndarray) -> np.ndarray:
        
        
        rot_matrix_z = np.array([[np.cos(rot_xyz[2]),  -np.sin(rot_xyz[2]), 0],
                                 [np.sin(rot_xyz[2]), np.cos(rot_xyz[2]),  0],
                                 [0,                 0,                 1]])
     
        rot_matrix_y = np.array([[np.cos(rot_xyz[1]), 0, 0],
                                 [0,                1,  0],
                                 [0,                0, 1]])
        
        rot_matrix_x = np.array([[1,  0,                0],
                                 [0, np.cos(rot_xyz[0]), 0],
                                 [0, 0,                1]])
        
        return rot_matrix_x@rot_matrix_y@rot_matrix_z
        
    def _get_shear_matrix(self, shear : np.ndarray) -> np.ndarray:
                
        shear_matrix = np.array([[1,       shear[0], 0],
                                 [shear[1], 1,       0],
                                 [0,      0,      1]])
        
        return shear_matrix
    
    def _get_scale_matrix(self, scale : np.ndarray) -> np.ndarray:
                
        scale_matrix = np.array([[scale[0], 0,      0],
                                 [0,      scale[1], 0],
                                 [0,      0,      1]])
    
        return scale_matrix

def random_homography(rot_xyz_range : np.ndarray =
                        np.array([(-np.pi/4, np.pi/4), (-np.pi/4, np.pi/4), (-np.pi, np.pi)]),
                      shear_range : np.ndarray = 
                        np.array([(-0.5, 0.5), (-0.5, 0.5)]),
                      scale_range : np.ndarray = 
                        np.array([(0.8, 1.2), (0.8, 1.2)]),
                      shift_range : np.ndarray = 
                        np.array([(-0.2, 0.2), (-0.2, 0.2)])):

    rot_xyz = np.random.rand(3)*(rot_xyz_range[:,1] - rot_xyz_range[:,0]) + rot_xyz_range[:,0]
    shear = np.random.rand(2)*(shear_range[:,1] - shear_range[:,0]) + shear_range[:,0]
    scale = np.random.rand(2)*(scale_range[:,1] - scale_range[:,0]) + scale_range[:,0]
    shift = np.random.rand(2)*(shift_range[:,1] - shift_range[:,0]) + shift_range[:,0]
    
    return Homography(rot_xyz, shear, scale, shift)


def perspective_transform(image : np.ndarray,
                          label : np.ndarray,
                          fill : list[tuple] = None):

    homographies = random_homography()

    # Extract the corners of the images (same for labels)
    corners = extract_corners(image)
    corners_new = homographies(corners)
    
    corners = project_to_cam(corners).T
    corners_new = project_to_cam(corners_new).T

    image = torchvision.transforms.functional.perspective(torch.tensor(image).permute(2,0,1),
                                                                corners.tolist(),
                                                                corners_new.tolist(),
                                                                fill = (255,255,255)).permute(1,2,0).numpy()
    label = torchvision.transforms.functional.perspective(torch.tensor(label).permute(2,0,1),
                                                                corners.tolist(),
                                                                corners_new.tolist(),
                                                                fill = (0,0,0)).permute(1,2,0).numpy()

    return image, label
    
class ImagePreProcessor:

    def __init__(self, image_size = 512):

        self.image_size = image_size

    def __call__(self, images, labels):

        images, labels = self._resize_and_pad(images, labels)

        return images, labels
    
    def _resize_and_pad(self, images, labels):
        # Scale the images and labels in a batch fashion where the final result should be a self.image_size x self.image_size picture. Do not change aspect ration. Instead pad with black
        
        padded_images = []
        padded_labels = []
        for image, label in zip(images, labels):
            h, w = image.shape[:2]
            max_size = max(h, w)
            pad_h = (max_size - h) // 2
            pad_w = (max_size - w) // 2

            padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(255,255,255))
            padded_label = cv2.copyMakeBorder(label, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
            padded_images.append(padded_image)
            padded_labels.append(padded_label)

        resized_images = []
        resized_labels = []
        for image, label in zip(padded_images, padded_labels):
            resized_image = cv2.resize(image, (self.image_size, self.image_size))
            resized_label = cv2.resize(label, (self.image_size, self.image_size))
            resized_images.append(resized_image)
            resized_labels.append(resized_label)
        return resized_images, resized_labels

class Augmentor:

    def __init__(self,
                 augment_prop = 0.5,
                 flip_prop = 0.3,
                 noise_prop = 0.6,
                 noise_amount = 0.2,
                 perspective_prop = 0.6,
                 cutout_param = None):

        self.augment_prop = augment_prop
        self.flip_prop = flip_prop
        self.noise_prop = noise_prop
        self.noise_amount = noise_amount
        self.perspective_prop = perspective_prop

        if cutout_param == None:
            cutout_param = {'min_size': 0.1, 'max_size': 0.2, 'n_cutouts': 3, 'cutout_prob': 0.2}
        self.cutout_param = cutout_param # (min_size, max_size, n_cutouts)

    def __call__(self, image, label):

        if random.random() > self.augment_prop:
            return image, label

        image, label = self._random_noise(image, label)
        image, label = self._random_flip(image, label)
        image, label = self._random_cutout(image, label)
        image, label = self._random_perspective(image, label)

        return image, label
    
    def _random_flip(self, image, label):

        if random.random() < self.flip_prop:
            image = cv2.flip(image, 1)
            image = cv2.flip(label, 1)
        if random.random() < self.flip_prop:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)

        return image, label
    
    def _random_cutout(self, image, label):

        if random.random() > self.cutout_param["cutout_prob"]:
            return image, label


        n_cutouts = random.randint(1, self.cutout_param["n_cutouts"])

        for _ in range(n_cutouts): # Number of cutouts
            w = random.random() * (self.cutout_param["max_size"] -
                                   self.cutout_param["min_size"]) + self.cutout_param["min_size"]
            h = random.random() * (self.cutout_param["max_size"] -
                                   self.cutout_param["min_size"]) + self.cutout_param["min_size"]

            h = int(h * image.shape[0])
            w = int(w * image.shape[1])

            image = cv2.rectangle(image, (h, w), (h + h, w + w), (0,0,0), -1)
            label = cv2.rectangle(label, (h, w), (h + h, w + w), (0,0,0), -1)

        return image, label
    
    def _random_perspective(self, image, label):
        
        if random.random() > self.perspective_prop:
            return image, label

        return perspective_transform(image, label)

    def _random_noise(self, image, label):

        if random.random() > self.noise_prop:
            return image, label

        noise = np.random.normal(0, self.noise_amount, image.shape)
        image = image + noise

        return image, label
