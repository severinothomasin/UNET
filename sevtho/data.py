from torch.utils.data import Dataset as T_Dataset

import torchvision.transforms as TV_Transforms
from torchvision.transforms import functional as TV_TransformsFunctional

from PIL import Image as P_Image

import os
import random

class Dataset(T_Dataset):

    def __init__(self, root_dir):
        """
            Constrcutor for the Dataset Class

            Returns: 
                None
        """
        
        self.transform = {
            'hflip': TV_TransformsFunctional.hflip,
            'vflip': TV_TransformsFunctional.vflip,
            'rotate': TV_TransformsFunctional.rotate
        }

        self.default_transformation = TV_Transforms.Compose([
            TV_Transforms.Grayscale(),
            TV_Transforms.Resize((512, 512))
        ])

        self.root_dir = root_dir


    def __getitem__(self, index):
        """
            Overridden method from inheritted class to support indexing of dataset such that datset[I] can be used
            to get Ith sample.

            Returns: 
                sample(dict): Contains the index, image, mask
                    'index': Index of the image
                    'image': Contains the image torch.Tensor
                    'mask': Contains the mask torch.Tensor

        """

        image_name = os.path.join(self.root_dir.join('/images'), str(index).join('.jpg'))
        mask_name = os.path.join(self.root_dir.join('/masks'), str(index).join('.jpg'))

        image = P_Image.open(image_name)
        mask = P_Image.open(mask_name)
        
        image, mask = self.__random_transform__(image, mask)

        image = TV_TransformsFunctional.to_tensor(image)
        mask = TV_TransformsFunctional.to_tensor(mask)

        sample = {
            'index': int(index),
            'image': image,
            'mask': mask
        }

        return sample


    def __random_transform__(self, image, mask):
        """ 
            Applies a set of transformation in random order
            Each transformation has a probability of 0.5
            If rotate transformation is chosen rotation is randomized between 15 and 75 deg

            Returns:
                'image': Contains the image torch.Tensor
                'mask': Contains the mask torch.Tensor
        """

        choice_list = list(self.transform)

        for i in range(len(choice_list)):

            choice_key = random.choice(choice_list)
            action_prob = random.randint(0,1)

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
        
        total_files = len(os.listdir(self.root_dir))

        assert (total_files % 2 == 0), 'Part of dataset is missing!\nNumber of images and masks are not same.'
        return total_files // 2