import os
import pickle
import numpy
import random

from torch.utils.data import Dataset
import torchvision.transforms as tt
import torchvision.transforms.functional as TF

from PIL import Image

import UNET_PARAMETERS as param

class Dataset():

    def __init__(self, transform=True):

        if not transform:
            self.transform = None 
        else:
            self.transform = {'hflip': TF.hflip,
                        'vflip': TF.vflip,
                        'rotate': TF.rotate}
            
        self.default_transformation = tt.Compose([
            tt.Grayscale(),
            tt.Resize((512, 512))
        ])

        self.dataset_name = param.DATASET_NAME
        self.dataset_path = os.path.join('dataset', self.dataset_name)
        self.images_path = os.path.join(self.dataset_path, 'images')
        self.masks_path = os.path.join(self.dataset_path, 'masks')

        self.test_split = param.TEST_SPLIT
        self.predict = param.PREDICT


    def __len__(self):

        len_images = len(os.listdir(self.images_path))
        
        if self.predict:
            return len_images
        else:
            len_masks = len(os.listdir(self.masks_path))
            total_files = len_masks + len_images
            assert (total_files %2 == 0), 'Number of Images and Masks is not the same'
            return total_files // 2


    def __getitem__(self, index):

        image_path = os.path.join(self.images_path, f'{index}.png')
        image = Image.open(image_path)
        image = self.default_transformation(image)
        
        if self.predict:
            image = TF.to_tensor(image)
            sample = {'index': int(index), 'image': image}
            return sample
        else:
            mask_path = os.path.join(self.masks_path, f'{index}_mask.png')
            mask = Image.open(mask_path)
            mask = self.default_transformation(mask)

            if self.transform:
                 image, mask = self.__random_transform__(image, mask)

            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)

            sample = {'index': int(index), 'image': image, 'mask': mask}
            return sample


    def __random_transform__(self, image, mask):
        
        choice_list = list(self.transform)

        for choice_id in range(len(choice_list)):

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

    def create_new_index(self):
        
        file_path = os.path.join('index_files', f'{self.dataset_name}.p')

        dataset_length = len(self)

        ids = list(range(dataset_length))
        numpy.random.shuffle(ids)

        with open(file_path, 'wb') as file:

            if self.predict:
                data = dict()
                data['image_ids'] = ids
            else:
                split = int(numpy.floor(self.test_split * dataset_length))
                data = dict()
                data['training_ids'] = ids[split:]
                data['testing_ids'] = ids[:split]
            
            pickle.dump(data, file)

    def get_ids(self):

        file_path = os.path.join('index_files', f'{self.dataset_name}.p')

        data = dict()

        if os.path.isfile(file_path):
            with open(file_path,'rb') as file:
                data = pickle.load(file)
        else:
            print('Couldnt find index file')
            return None

        if self.predict:
            return data['image_ids']
        else:
            return data['training_ids'], data['testing_ids']

