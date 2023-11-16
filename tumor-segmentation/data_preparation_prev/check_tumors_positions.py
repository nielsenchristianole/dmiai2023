import os
from PIL import Image

def find_extreme_segment_heights(directory):
    lowest_y_global, highest_y_global = float('inf'), float('-inf')

    for filename in os.listdir(directory):
        if filename.lower().endswith('.png'):
            image_path = os.path.join(directory, filename)
            with Image.open(image_path) as img:
                width, height = img.size
                lowest_y, highest_y = height, 0

                for y in range(height):
                    for x in range(width):
                        if img.getpixel((x, y)) == (255, 255, 255, 255):  
                            lowest_y = min(lowest_y, y)
                            highest_y = max(highest_y, y)

                # Update global extremes if a segment was found in this image
                if lowest_y < highest_y:
                    lowest_y_global = min(lowest_y_global, lowest_y)
                    highest_y_global = max(highest_y_global, highest_y)

    return lowest_y_global, highest_y_global

extreme_heights = find_extreme_segment_heights('/Users/aleksandra/Desktop/resized_labels_NN')
print(extreme_heights)


