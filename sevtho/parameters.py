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
BATCH_SIZE = 1
# Dataset folder used
DATASET_NAME = 'm1E8'
# Training Epochs
EPOCHS = 100
# Filters used in UNet Model
FILTER_LIST = [16,32,64,128,256]
# Flag to train the model
TRAIN = False
# Flag to load saved model
LOAD_MODEL = True
# Flag to save model trained
SAVE_MODEL = False
# Flag to save the file index
NEW_INDEX = True
# Apply Model to fresh Dataset
APPLY_DATASET = True
# Model Name
MODEL_NAME = 'm1E8'