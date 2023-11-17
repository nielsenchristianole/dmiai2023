#!/bin/bash

#resize img and labels
echo "I am changing size of patiens and labels images."
python data_preparation/change_size.py

echo "I am pairing control images with labels based on size and adjusting the size before creating synthetic data."
#pair the control with labels and resize to create synthetic data 
python data_preparation/synthetic_data.py

echo "I am creating syntetic data"
#plotting labels with control data
python data_preparation/plot_segments.py

echo "I am resizing the synthetic data"
#resize synthetic data
python data_preparation/change_size_synthetic.py

echo "I am combining all images"
#combine all images and labels
python data_preparation/combine.py

echo "I am spliting all images"
#split the dataset
python data_preparation/data_split.py

echo "I am training now. :)"
#train U-Net
python train.py
