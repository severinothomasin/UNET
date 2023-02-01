# -*- coding: utf-8 -*-
"""UNET Implementation in Pytorch

@Author: Severino Thomasin
@Date: 30.01.2023
@Links: https://www.mls.uzh.ch/en/research/stoeckli
"""

"""
This script contains all the necessary parameters for the algorithm. 
Modify it according to your data.

In Case of Questions, write an email to severino.thomasin@uzh.ch
"""

# How much of your dataset should be used for testing (%)
TEST_SPLIT = 0.2
# Batch size for training. Limited by GPU memory
BATCH_SIZE = 6
# Training Epochs
EPOCHS = 5

# Name of the Model 
MODEL_NAME = 'brain_tumors'
# Name of the Dataset
DATASET_NAME = 'brain_tumors'

# Predict Mask on new dataset based on preexisting model without training the preexisting model
PREDICT = False
# Create new index for dataset or use preexisting one
NEW_INDEX = True
# Filters used in UNet Model
FILTER_LIST = [16,32,64,128,256]
# Load a preexisting model or create a new one
LOAD_MODEL = False
# Save Changes made to model 
SAVE_MODEL = True
# Train the model based on given dataset
TRAIN_MODEL = True
