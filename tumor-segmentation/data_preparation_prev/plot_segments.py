import cv2
import os

def apply_masks_to_images(images_folder, masks_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(images_folder):
        image_path = os.path.join(images_folder, filename)
        mask_path = os.path.join(masks_folder, filename)

        if os.path.exists(mask_path):
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            mask_inv = cv2.bitwise_not(mask)
            image[mask_inv == 0] = [0, 0, 0]

            cv2.imwrite(os.path.join(output_folder, filename), image)
        else:
            print(f"Lack of mask for{filename}")

images_folder = 'data/synthetic_images'
masks_folder = 'data/synthetic_mask'
output_folder = 'data/synthetic_images_plotted'
apply_masks_to_images(images_folder, masks_folder, output_folder)
