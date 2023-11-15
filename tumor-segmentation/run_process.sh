#!/bin/bash

#resize img and labels
echo "I am changing size of patiens and labels images."
python3 ./tumor-segmentation/data_preparation/change_size.py

echo "I am pairing control images with labels based on size and adjust the size before creating synthetic data."
#pair the control with labels and resize to create synthetic data 
python3 ./tumor-segmentation/data_preparation/synthetic_data.py

echo "I am creating syntetic data"
#plotting labels with control data
python3 ./tumor-segmentation/data_preparation/plot_segments.py

echo "I am resizing the synthetic data"
#resize synthetic data
python3 ./tumor-segmentation/data_preparation/change_size_synthetic.py

echo "I am combining all images"
#combine all images and labels
python3 ./tumor-segmentation/data_preparation/combine.py

echo "I spliting all images"
#split the dataset
python3 ./tumor-segmentation/data_preparation/data_split.py

#train U-Net
python3 ./tumor-segmentation/train.py
