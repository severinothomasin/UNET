import os

import torch
from torch.utils.data import SubsetRandomSampler, DataLoader

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import sevtho.dataset
import sevtho.model 
import sevtho.classifier

import UNET_PARAMETERS as param


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = sevtho.dataset.Dataset()

if param.NEW_INDEX:

    dataset.create_new_index()


if param.PREDICT:

    image_ids = dataset.get_ids()
    image_sampler = SubsetRandomSampler(image_ids)
    image_loader = DataLoader(dataset, param.BATCH_SIZE, sampler=image_sampler)

    unet_model = sevtho.model.DynamicUNet(param.FILTER_LIST)
    unet_classifier = sevtho.classifier.Classifier(unet_model, device)
    unet_classifier.restore_model(os.path.join('saved_models',f'{param.MODEL_NAME}.pt'))

    for i in range(0, 5):
        sample = dataset[image_ids[i]]
        image, output = unet_classifier.predict(sample, 0.65)

        title = f'Name: {image_ids[i]}.png'
        transparency = 0.38

        seg_output = output * transparency
        seg_image = np.add(image, seg_output) / 2

        f, axarr = plt.subplots(nrows=1,ncols=3,sharex=True, sharey=True, figsize=(20, 15), gridspec_kw={'wspace': 0.025, 'hspace': 0.010})
        f.suptitle(title, x=0.5, y=0.92, fontsize=20)

        plt.sca(axarr[0]); 
        plt.imshow(image, cmap='gray'); 
        plt.title("Original Image", fontdict={'fontsize': 16})
        plt.sca(axarr[1]); 
        plt.imshow(output, cmap='Blues'); 
        plt.title("Constructed Mask", fontdict={'fontsize': 16})
        plt.sca(axarr[2]); 
        plt.imshow(seg_image, cmap='gray'); 
        plt.title("Constructed Segment", fontdict={'fontsize': 16})
        plt.show()

else:

    training_ids, testing_ids = dataset.get_ids()

    training_sampler = SubsetRandomSampler(training_ids)
    training_loader = DataLoader(dataset, param.BATCH_SIZE, sampler=training_sampler)

    testing_sampler = SubsetRandomSampler(testing_ids)
    testing_loader = DataLoader(dataset, 1, sampler=testing_sampler)

    if param.LOAD_MODEL:
        unet_model = sevtho.model.DynamicUNet(param.FILTER_LIST)
        unet_classifier = sevtho.classifier.Classifier(unet_model, device)
        unet_classifier.restore_model(os.path.join('saved_models',f'{param.MODEL_NAME}.pt'))
        print(f'Saved model {param.MODEL_NAME} loaded')         
    else:
        unet_model = sevtho.model.DynamicUNet(param.FILTER_LIST).to(device)
        unet_classifier = sevtho.classifier.Classifier(unet_model, device)
        print(f'New model {param.MODEL_NAME} created')   

    if param.TRAIN_MODEL:
        unet_model.train()

        path = None
        if param.SAVE_MODEL:
            path = os.path.join('saved_models',f'{param.MODEL_NAME}.pt')

        unet_train_history = unet_classifier.train(param.EPOCHS, training_loader, mini_batch=100, save_best=path)
        print(f'Training Finished after {param.EPOCHS} epoches')

        save_plot = os.path.join('images',f'UNet-{param.MODEL_NAME}-loss_graph.png')        
        plt.figure(figsize=(20, 10))
        plt.title('Loss Function Over Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        line = plt.plot(unet_train_history['train_loss'], marker='o')
        plt.legend((line), ('Loss Value',), loc=1)
        plt.savefig(save_plot)
        plt.show()
    
    i=0

    while True:
        image_index = testing_ids[i]
        sample = dataset[image_index]
        image, mask, output, d_score = unet_classifier.predict(sample,0.65)
        title = f'Name: {image_index}.png   Dice Score: {d_score:.5f}'
        # save_path = os.path.join('images',f'{d_score:.5f}_{image_index}.png')
        transparency=0.38
        fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(
            20, 15), gridspec_kw={'wspace': 0.025, 'hspace': 0.010})
        fig.suptitle(title, x=0.5, y=0.92, fontsize=20)

        axs[0][0].set_title("Original Mask", fontdict={'fontsize': 16})
        axs[0][0].imshow(mask, cmap='gray')
        axs[0][0].set_axis_off()

        axs[0][1].set_title("Constructed Mask", fontdict={'fontsize': 16})
        axs[0][1].imshow(output, cmap='gray')
        axs[0][1].set_axis_off()

        mask_diff = np.abs(np.subtract(mask, output))
        axs[0][2].set_title("Mask Difference", fontdict={'fontsize': 16})
        axs[0][2].imshow(mask_diff, cmap='gray')
        axs[0][2].set_axis_off()

        seg_output = mask*transparency
        seg_image = np.add(image, seg_output)/2
        axs[1][0].set_title("Original Segment", fontdict={'fontsize': 16})
        axs[1][0].imshow(seg_image, cmap='gray')
        axs[1][0].set_axis_off()

        seg_output = output*transparency
        seg_image = np.add(image, seg_output)/2
        axs[1][1].set_title("Constructed Segment", fontdict={'fontsize': 16})
        axs[1][1].imshow(seg_image, cmap='gray')
        axs[1][1].set_axis_off()

        axs[1][2].set_title("Original Image", fontdict={'fontsize': 16})
        axs[1][2].imshow(image, cmap='gray')
        axs[1][2].set_axis_off()

        plt.tight_layout()

        save_path = os.path.join('images/examples', param.DATASET_NAME)
        plt.savefig(save_path, dpi=90, bbox_inches='tight')

        plt.show()        
        i += 1
        if i >= len(testing_ids):
            i = 0 

