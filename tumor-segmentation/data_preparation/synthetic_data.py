#chmod +x tumor-segmentation/run_process.sh

import os
from PIL import Image

def get_image_sizes_with_filenames(directory):
    sizes_with_filenames = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.bmp')):
            image_path = os.path.join(directory, filename)
            with Image.open(image_path) as img:
                size = img.size
                if size not in sizes_with_filenames:
                    sizes_with_filenames[size] = []
                sizes_with_filenames[size].append(filename)
    return sizes_with_filenames

def match_closest_sizes(sizes_with_filenames1, sizes_with_filenames2):
    matched_pairs = []
    for size1, filenames1 in sizes_with_filenames1.items():
        closest_size = None
        min_diff = float('inf')
        for size2 in sizes_with_filenames2.keys():
            height_diff = abs(size1[1] - size2[1])
            if height_diff <= 80:
                diff = abs(size1[0] * size1[1] - size2[0] * size2[1])
                if diff < min_diff:
                    min_diff = diff
                    closest_size = size2
        if closest_size:
            matched_pairs.append((size1, closest_size, filenames1, sizes_with_filenames2[closest_size]))
    return matched_pairs


# directories
dir1 = './tumor-segmentation/data/controls/imgs'
dir2 = './tumor-segmentation/data/patients/labels'
save_dir1 = './tumor-segmentation/data/synthetic_images'
save_dir2 = './tumor-segmentation/data/synthetic_mask'

#create
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def resize_and_save_images(matched_pairs, dir1, dir2, save_dir1, save_dir2):
    ensure_directory_exists(save_dir1)
    ensure_directory_exists(save_dir2)



sizes_with_filenames_dir1 = get_image_sizes_with_filenames(dir1)
sizes_with_filenames_dir2 = get_image_sizes_with_filenames(dir2)
matched_pairs = match_closest_sizes(sizes_with_filenames_dir1, sizes_with_filenames_dir2)

for size1, size2, filenames1, filenames2 in matched_pairs:
    print(f"Size Pair: {size1} and {size2}")
    print(f"Files from Directory 1: {filenames1}")
    print(f"Files from Directory 2: {filenames2}")


def resize_and_save_images(matched_pairs, dir1, dir2, save_dir1, save_dir2):
    ensure_directory_exists(save_dir1)
    ensure_directory_exists(save_dir2)
    image_counter = 1  # Counter for each processed image pair

    for size1, size2, filenames1, filenames2 in matched_pairs:
        for filename1, filename2 in zip(filenames1, filenames2):
            # Process and save image from dir1
            original_path1 = os.path.join(dir1, filename1)
            save_path1 = os.path.join(save_dir1, f'synthetic_{image_counter}.png')
            with Image.open(original_path1) as img:
                img.save(save_path1)

            # Resize and save image from dir2
            image_path2 = os.path.join(dir2, filename2)
            with Image.open(image_path2) as img:
                if img.size[1] > size1[1]:  # Crop if larger
                    img_resized = img.crop((0, 0, img.size[0], size1[1]))
                else:  # Add black pixels if smaller
                    img_resized = Image.new('RGB', (size1[0], size1[1]), (0, 0, 0))
                    img_resized.paste(img, (0, 0))

                resized_path = os.path.join(save_dir2, f'synthetic_{image_counter}.png')
                img_resized.save(resized_path)

            image_counter += 1

sizes_with_filenames_dir1 = get_image_sizes_with_filenames(dir1)
sizes_with_filenames_dir2 = get_image_sizes_with_filenames(dir2)
matched_pairs = match_closest_sizes(sizes_with_filenames_dir1, sizes_with_filenames_dir2)
resize_and_save_images(matched_pairs, dir1, dir2, save_dir1, save_dir2)