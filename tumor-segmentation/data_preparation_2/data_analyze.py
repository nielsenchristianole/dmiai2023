# import os
# from PIL import Image

# def print_unique_image_sizes(directory):
#     unique_sizes = set()
#     smallest_image = None
#     largest_image = None
#     smallest_size = (float('inf'), float('inf'))  # Initialize with infinity
#     largest_size = (0, 0)  # Initialize with zero

#     for filename in os.listdir(directory):
#         if filename.lower().endswith(('.png','.bmp')):
#             image_path = os.path.join(directory, filename)
#             with Image.open(image_path) as img:
#                 size = img.size
#                 unique_sizes.add(size)

#                 # Update smallest image
#                 if size[0] * size[1] < smallest_size[0] * smallest_size[1]:
#                     smallest_size = size
#                     smallest_image = filename

#                 # Update largest image
#                 if size[0] * size[1] > largest_size[0] * largest_size[1]:
#                     largest_size = size
#                     largest_image = filename

#     print("Unique image sizes:", unique_sizes)
#     if smallest_image:
#         print("Smallest image:", smallest_image, "with dimensions", smallest_size)
#     if largest_image:
#         print("Largest image:", largest_image, "with dimensions", largest_size)

# print_unique_image_sizes('/Users/aleksandra/Desktop/Champ_AI/DM-i-AI-2023/tumor-segmentation/data/controls/imgs')
# print_unique_image_sizes('/Users/aleksandra/Desktop/Champ_AI/DM-i-AI-2023/tumor-segmentation/data/patients/imgs')


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

dir1 = '/Users/aleksandra/Desktop/Champ_AI/DM-i-AI-2023/tumor-segmentation/data/controls/imgs'
dir2 = '/Users/aleksandra/Desktop/Champ_AI/DM-i-AI-2023/tumor-segmentation/data/patients/imgs'

sizes_with_filenames_dir1 = get_image_sizes_with_filenames(dir1)
sizes_with_filenames_dir2 = get_image_sizes_with_filenames(dir2)
matched_pairs = match_closest_sizes(sizes_with_filenames_dir1, sizes_with_filenames_dir2)

for size1, size2, filenames1, filenames2 in matched_pairs:
    print(f"Size Pair: {size1} and {size2}")
    print(f"Files from Directory 1: {filenames1}")
    print(f"Files from Directory 2: {filenames2}")