import os
import glob

import numpy as np
from PIL import Image


THRESHHOLD = 0.102
LABEL_DIR = '../data/patients/labels'


class BaselineModel():

    def __init__(
        self,
        threashhold: float=THRESHHOLD,
        label_dir: str=LABEL_DIR,
    ):
        label_paths = glob.glob(os.path.join(label_dir, '*.png'))
        
        masks = list()
        for p in label_paths:
            img = Image.open(p).convert('L')
            resized_img = img.resize((400, 400))
            masks.append(
                np.array(resized_img)[None, ...] / 255.
            )
        masks = np.concatenate(masks, axis=0)
        self.mask = Image.fromarray(
            masks.mean(axis=0) >= threashhold
        )
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        out_shape = image.shape[1], image.shape[0]
        grayscale = 255 * np.array(
            self.mask.resize(
                out_shape,
                Image.NEAREST
            ),
            dtype=np.uint8
        )
        
        return np.stack((grayscale, grayscale, grayscale), axis=-1)


