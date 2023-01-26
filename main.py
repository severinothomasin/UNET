import torch
import os
import numpy as np
import pickle

from torch.utils.data import SubsetRandomSampler as T_SubsetRandomSampler
from torch.utils.data import DataLoader as T_DataLoader

import sevtho.data

# --------------------------------------
TEST_SPLIT = 0.2
BATCH_SIZE = 4
DATASET_PATH = os.path.join('data','train')
EPOCHS = 100
FILTER_LIST = [16,32,64,128,256]
LOAD_MODEL = False
# --------------------------------------

def get_ids(length, new=False):
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
train_ids, test_ids = get_ids(len(dataset))
train_sampler, test_sampler = T_SubsetRandomSampler(train_ids), T_SubsetRandomSampler(test_ids)

train_loader = T_DataLoader(dataset, BATCH_SIZE, sampler=train_sampler)
test_loader = T_DataLoader(dataset, 1, sampler=test_sampler)

