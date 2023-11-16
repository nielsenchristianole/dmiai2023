#!/bin/bash

#resize img and labels
echo "I am changing size of patiens and labels images."
python3 data_preparation_2/change_size.py

echo "I am pairing control images with labels based on size and adjusting the size before creating synthetic data."
#pair the control with labels and resize to create synthetic data 
python3 data_preparation_2/synthetic_data.py

echo "I am creating syntetic data"
#plotting labels with control data
python3 data_preparation_2/plot_segments.py

echo "I am resizing the synthetic data"
#resize synthetic data
python3 data_preparation_2/change_size_synthetic.py

echo "I am combining all images"
#combine all images and labels
python3 data_preparation_2/combine.py

echo "I am spliting all images"
#split the dataset
python3 data_preparation_2/data_split.py

echo "I am training now. :)"
#train U-Net
python3 train.py
