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
TEST_SPLIT = 0.3
# Batch size for training. Limited by GPU memory
BATCH_SIZE = 6
# Dataset folder used
DATASET_NAME = 'brain_tumors'
# Training Epochs
EPOCHS = 5000
# Filters used in UNet Model
FILTER_LIST = [16,32,64,128,256]
# Flag to train the model
TRAIN = True
# Flag to load saved model
LOAD_MODEL = False
# Flag to save model trained
SAVE_MODEL = True
# Flag to save the file index
SAVE_INDEX = True