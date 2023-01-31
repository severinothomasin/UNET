import os

import torch
from torch.utils.data import SubsetRandomSampler, DataLoader

from tabulate import tabulate

import sevtho.parameters as parameters 
import sevtho.dataset as dataset
import sevtho.model as model
import sevtho.classifier as classifier
import sevtho.plot as plot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device Used: {device}')
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
    ['SAVE_INDEX', parameters.NEW_INDEX], 
    ['BATCH_SIZE', parameters.BATCH_SIZE]], headers=['Parameter', 'Value']))
print('\n')

model_name = f"{parameters.DATASET_NAME}.pt"

tumor_dataset = dataset.Dataset()

if parameters.NEW_INDEX:
    tumor_dataset.create_new_index()

if parameters.APPLY_DATASET:
    image_ids = tumor_dataset.get_ids()
else:
    training_ids, testing_ids = tumor_dataset.get_ids()

training_sampler = SubsetRandomSampler(training_ids)
testing_sampler = SubsetRandomSampler(testing_ids)

trainloader = DataLoader(tumor_dataset, parameters.BATCH_SIZE, sampler=training_sampler)
testloader = DataLoader(tumor_dataset, 1, sampler=testing_sampler)

if not parameters.LOAD_MODEL:
    unet_model = model.DynamicUNet(parameters.FILTER_LIST).to(device)
    unet_classifier = classifier.BrainTumorClassifier(unet_model,device)
else:
    # Saved model is loaded on memory.
    unet_model = model.DynamicUNet(parameters.FILTER_LIST)
    unet_classifier = classifier.BrainTumorClassifier(unet_model,device)
    unet_classifier.restore_model(os.path.join('saved_models',model_name))
    print('Saved model loaded')    

if parameters.TRAIN:
    unet_model.train()
    path = os.path.join('saved_models',model_name) if parameters.SAVE_MODEL else None
    unet_train_history = unet_classifier.train(parameters.EPOCHS,trainloader,mini_batch=100,save_best=path)
    print(f'Training Finished after {parameters.EPOCHS} epoches')

# Testing process on test data.
unet_model.eval()
unet_score = unet_classifier.test(testloader)
print(f'\n\nDice Score {unet_score}')

if parameters.TRAIN:
    save_plot = os.path.join('images',f'UNet-{parameters.FILTER_LIST}-loss_graph.png')
    plot.loss_graph(unet_train_history['train_loss'],save_plot)

for i in range(0,4):
    image_index = training_ids[i]
    sample = tumor_dataset[image_index]
    #image, mask, output, d_score = unet_classifier.predict(sample,0.65)
    image, output = unet_classifier.predict(sample,0.65)
    #title = f'Name: {image_index}.png   Dice Score: {d_score:.5f}'
    title = f'Name: {image_index}.png'
    # save_path = os.path.join('images',f'{d_score:.5f}_{image_index}.png')
    #plot.result(image,mask,output,title,save_path=None)
    plot.result(image,output,title,save_path=None)