import torch
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from datetime import datetime
from time import time

import sevtho.loss

import UNET_PARAMETERS as param

class Classifier():

    def __init__(self, model, device):

        self.model = model
        self.device = device
        self.optimizer = None
        
        self.criterion = sevtho.loss.BCEDiceLoss(self.device).to(self.device)
    

    def train(self, epochs, trainloader, mini_batch=None, learning_rate=0.001, save_best=None):

        history = {'train_loss': list()}

        last_loss = 1000

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Reducing LR on plateau feature to improve training.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.85, patience=2, verbose=True)

        print('Starting Training Process')

        for epoch in range(epochs):
            start_time = time()
            epoch_loss = self.__train_epoch__(trainloader, mini_batch)
            history['train_loss'].append(epoch_loss)
            self.scheduler.step(epoch_loss)

            time_taken = time() - start_time

            print(f'Epoch: {epoch+1:03d},  ', end='')
            print(f'Loss:{epoch_loss:.7f},  ', end='')
            print(f'Time:{time_taken:.2f}secs', end='')

            if save_best != None and last_loss > epoch_loss:
                self.__save_model__(save_best)
                last_loss = epoch_loss
                print(f'\tSaved at loss: {epoch_loss:.10f}')
            else:
                print()

        return history

    
    def __train_epoch__(self, dataloader, mini_batch):

        epoch_loss, batch_loss, batch_iteration = 0, 0, 0

        for batch, data in enumerate(dataloader):
            batch_iteration += 1

            image = data['image'].to(self.device)
            mask = data['mask'].to(self.device)

            self.optimizer.zero_grad()

            output = self.model(image)
            loss_value = self.criterion(output, mask)
            loss_value.backward()

            self.optimizer.step()
            epoch_loss += loss_value.item()
            batch_loss += loss_value.item()

            if mini_batch:
                if (batch + 1) % mini_batch == 0:
                    batch_loss = batch_loss / (mini_batch * dataloader.batch_size)
                    print(f'Batch: {batch+1:02d},\tBatch Loss: {batch_loss:.7f}')
                    batch_loss = 0

        epoch_loss = epoch_loss / (batch_iteration * dataloader.batch_size)
        return epoch_loss        


    def __save_model__(self, path):
        torch.save(self.model.state_dict(), path)

    
    def restore_model(self, path):
        if self.device == 'cpu':
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            self.model.load_state_dict(torch.load(path))
            self.model.to(self.device)

    
    def predict(self, data, threshold=0.5):

        self.model.eval()

        image = data['image'].numpy()

        image_tensor = torch.Tensor(data['image'])
        image_tensor = image_tensor.view((-1, 1, 512, 512)).to(self.device)

        output = self.model(image_tensor).detach().cpu()
        output = (output > threshold)
        output = output.numpy()

        image = np.resize(image, (512, 512))
        output = np.resize(output, (512, 512))

        if param.PREDICT:
            return image, output
        else:
            mask = data['mask'].numpy()
            mask = np.resize(mask, (512, 512))
            score = self.__dice_coefficient__(output, mask)
            return image, mask, output, score
    
    def __dice_coefficient__(self, predicted, target):
        smooth = 1
        product = np.multiply(predicted, target)
        intersection = np.sum(product)
        coefficient = (2*intersection + smooth) / (np.sum(predicted) + np.sum(target) + smooth)
        return coefficient