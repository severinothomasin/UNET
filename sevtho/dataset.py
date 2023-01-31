import os
import pickle
import numpy
import random

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image

import sevtho.parameters as parameters

class Dataset():

    def __init__(self, transform=True):

        self.dataset_name = parameters.DATASET_NAME
        self.model_name = parameters.MODEL_NAME
        self.dataset_path = os.path.join('dataset', parameters.DATASET_NAME)
        self.image_folder = os.path.join(self.dataset_path,'images')
        self.mask_folder = os.path.join(self.dataset_path,'masks')
        self.test_split = parameters.TEST_SPLIT
        self.apply_dataset = parameters.APPLY_DATASET

        self.transform = {'hflip': TF.hflip,
                          'vflip': TF.vflip,
                          'rotate': TF.rotate}
        self.default_transformation = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((512, 512))
        ])

        if not transform:
            self.transform = None
    
    def __getitem__(self, index):
        """ Overridden method from inheritted class to support
        indexing of dataset such that datset[I] can be used
        to get Ith sample.
        Parameters:
            index(int): Index of the dataset sample

        Return:
            sample(dict): Contains the index, image, mask torch.Tensor.
                        'index': Index of the image.
                        'image': Contains the tumor image torch.Tensor.
                        'mask' : Contains the mask image torch.Tensor.
        """

        if not self.apply_dataset:
        
            image_dir = os.path.join(self.dataset_path, 'images')
            mask_dir = os.path.join(self.dataset_path, 'masks')
        
            image_name = os.path.join(image_dir, str(index)+'.png')
            mask_name = os.path.join(mask_dir, str(index)+'_mask.png')

            image = Image.open(image_name)
            mask = Image.open(mask_name)

            image = self.default_transformation(image)
            mask = self.default_transformation(mask)

            if self.transform:
                image, mask = self._random_transform(image, mask)

            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)

            sample = {'index': int(index), 'image': image, 'mask': mask}
            return sample
        
        else:

            image_dir = os.path.join(self.dataset_path, 'images')
            image_name = os.path.join(image_dir, str(index)+'.png')
            image = Image.open(image_name)
            image = TF.to_tensor(image)
            sample = {'index': int(index), 'image': image}
            return sample

    def _random_transform(self, image, mask):
        """ Applies a set of transformation in random order.
        Each transformation has a probability of 0.5
        """
        choice_list = list(self.transform)
        for _ in range(len(choice_list)):
            choice_key = random.choice(choice_list)
            action_prob = random.randint(0, 1)
            if action_prob >= 0.5:
                if choice_key == 'rotate':
                    rotation = random.randint(15, 75)
                    image = self.transform[choice_key](image, rotation)
                    mask = self.transform[choice_key](mask, rotation)
                else:
                    image = self.transform[choice_key](image)
                    mask = self.transform[choice_key](mask)
            choice_list.remove(choice_key)

        return image, mask

    def __len__(self):

        if not self.apply_dataset:
            total_files = len(os.listdir(self.image_folder)) + len(os.listdir(self.mask_folder)) 
            assert (total_files % 2 == 0), 'Number of Images and Masks is not the same'
            return total_files // 2
        
        else:
            return len(os.listdir(self.image_folder))


    def create_new_index(self):

        file_path = f'{self.dataset_path}_ids.p'
        
        data = dict()

        dataset_length = len(self)

        ids = list(range(dataset_length))
        numpy.random.shuffle(ids)
        split = int(numpy.floor(self.test_split * dataset_length))

        if not self.apply_dataset:
            data['training_ids'] = ids[split:]
            data['testing_ids'] = ids[:split]

            with open(file_path, 'wb') as file:
                pickle.dump(data, file)
        else:
            data['training_ids'] = ids

            with open(file_path, 'wb') as file:
                pickle.dump(data, file)


    def get_ids(self):

        file_path = f'{self.dataset_path}_ids.p'
        
        data = dict()

        if os.path.isfile(file_path):
            with open(file_path,'rb') as file:
                data = pickle.load(file)
        else:
            print('Couldnt find index file')
            return None

        if not self.apply_dataset:
            return data['training_ids'], data['testing_ids']
        else:
            return data['training_ids']