import torch
import os
import numpy as np
import pickle

from torch.utils.data import SubsetRandomSampler as T_SubsetRandomSampler
from torch.utils.data import DataLoader as T_DataLoader

import sevtho.data
import sevtho.model
import sevtho.classifier
import sevtho.plot

# --------------------------------------
TEST_SPLIT = 0.2
BATCH_SIZE = 4
DATASET_PATH = os.path.join('data','train')
EPOCHS = 100
FILTER_LIST = [16,32,64,128,256]
LOAD_MODEL = True
MODEL_NAME = f"UNet-{FILTER_LIST}.pt"
TRAIN = True
SAVE_MODEL = False
NEW_FILE_INDEX = False
# --------------------------------------

def get_ids(length, new):
    """
        Read out indices of training data and save it as a file so it can quickly be restored later on.

        Parameters:
            length(int): Length of dataset used
            dataset(sevtho.data.Dataset): Dataset used 
            new(boolean): If True, generate new pickle file with ids
        Returns:
            train_ids(list): Array of ids used for training 
            test_ids(list): Array of ids used for testing 
    """

    file_path = os.path.join(DATASET_PATH, 'split_ids.p')

    data = dict()
    train_ids, test_ids = [], []

    if os.path.isfile(file_path) and not new:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            train_ids, test_ids = data['train_ids'], data['test_ids']
    else:
        ids = list(range(length))
        np.random.shuffle(ids)
        split = int(np.floor(TEST_SPLIT * length))

        train_ids, test_ids = ids[split:], ids[:split]
        data['train_ids'], data['test_ids'] = train_ids, test_ids

        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
    
    return train_ids, test_ids

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device Used: ({device}) {torch.cuda.get_device_name(torch.cuda.current_device())}')
print(f'Pytorch Version: {torch.__version__}')

dataset = sevtho.data.Dataset(DATASET_PATH)
train_ids, test_ids = get_ids(len(dataset), NEW_FILE_INDEX)
train_sampler, test_sampler = T_SubsetRandomSampler(train_ids), T_SubsetRandomSampler(test_ids)

train_loader = T_DataLoader(dataset, BATCH_SIZE, sampler=train_sampler)
test_loader = T_DataLoader(dataset, 1, sampler=test_sampler)

print(vars(train_loader.sampler))
print(vars(test_loader.sampler))

#unet_model = None
#unet_classifier = None
#
#if not LOAD_MODEL:
#    unet_model = sevtho.model.DynamicUNet(FILTER_LIST).to(device)
#    print(unet_model.summary())
#    unet_classifier = sevtho.classifier.Classifier(unet_model, device)
#else:
#    unet_model = sevtho.model.DynamicUNet(FILTER_LIST)
#    unet_classifier = sevtho.classifier.Classifier(unet_model,device)
#    unet_classifier.restore_model(os.path.join('saved_models',MODEL_NAME))
#    print('Saved model loaded')
#    print(unet_model.summary())
#
#
# Training process
#if TRAIN:
#    unet_model.train()
#    path = os.path.join('saved_models',MODEL_NAME) if SAVE_MODEL else None
#    unet_train_history = unet_classifier.train(EPOCHS,train_loader,mini_batch=100,save_best=path)
#    print(f'Training Finished after {EPOCHS} epoches')
#
# Testing process on test data.
#unet_model.eval()
#unet_score = unet_classifier.test(test_loader)
#print(f'\n\nDice Score {unet_score}')
# Dice Score 0.7446110107881675
#
#save_plot = os.path.join('images',f'{MODEL_NAME}-loss_graph.png')
#sevtho.plot.loss_graph(unet_train_history['train_loss'],save_plot)
#
#for i in range(0,20):
#    # Run this cell repeatedly to see some results.
#    image_index = test_ids[i]
#    sample = dataset[image_index]
#    image, mask, output, d_score = unet_classifier.predict(sample,0.5)
#    title = f'Name: {image_index}.png   Dice Score: {d_score:.5f}'
#    # save_path = os.path.join('images',f'{d_score:.5f}_{image_index}.png')
#    sevtho.plot.result(image,mask,output,title,save_path=None)
#    i += 1
#    if i >= len(test_ids):
#        i = 0 
