import os

import torch
from torch.utils.data import SubsetRandomSampler, DataLoader

from tabulate import tabulate

import sevtho.parameters as parameters 
import sevtho.dataset as dataset
import sevtho.model as model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device Used: {torch.cuda.get_device_name(torch.cuda.current_device())}')
print(f'Cuda Version: {torch.version.cuda}')
print('\n')
print('Configurations:')
print('\n')
print(tabulate([
    ['TEST_SPLIT', parameters.TEST_SPLIT], 
    ['EPOCHS', parameters.EPOCHS], 
    ['FILTER_LIST', parameters.FILTER_LIST], 
    ['TRAIN', parameters.TRAIN], 
    ['LOAD_MODEL', parameters.LOAD_MODEL], 
    ['SAVE_MODEL', parameters.SAVE_MODEL], 
    ['SAVE_INDEX', parameters.SAVE_INDEX], 
    ['BATCH_SIZE', parameters.BATCH_SIZE]], headers=['Parameter', 'Value']))
print('\n')

tumor_dataset = dataset.Dataset()
training_ids, testing_ids = tumor_dataset.get_ids(new=False)

training_sampler = SubsetRandomSampler(training_ids)
testing_sampler = SubsetRandomSampler(testing_ids)

trainloader = DataLoader(tumor_dataset, parameters.BATCH_SIZE, sampler=training_sampler)
testloader = DataLoader(tumor_dataset, 1, sampler=testing_sampler)

if parameters.LOAD_MODEL == True:
    unet_model = model.DynamicUNet(parameters.FILTER_LIST)
    unet_classifier = classifier.