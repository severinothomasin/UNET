import os
import pickle
import numpy

import sevtho.parameters as parameters

class Dataset():

    def __init__(self):

        self.dataset_name = parameters.DATASET_NAME
        self.dataset_path = f'dataset/{parameters.DATASET_NAME}'
        self.image_folder = os.path.join(self.dataset_path,'images')
        self.mask_folder = os.path.join(self.dataset_path,'masks')
        self.test_split = parameters.TEST_SPLIT
        
    def __len__(self):

        total_files = len(os.listdir(self.image_folder)) + len(os.listdir(self.mask_folder)) 

        assert (total_files % 2 == 0), 'Number of Images and Masks is not the same'
        return total_files // 2

    def get_ids(self, new=True):

        file_path = f'{self.dataset_path}_ids.p'
        
        data = dict()

        dataset_length = len(self)

        if os.path.isfile(file_path) and not new:
            with open(file_path,'rb') as file:
                data = pickle.load(file)
        else:
            ids = list(range(dataset_length))
            numpy.random.shuffle(ids)
            split = int(numpy.floor(self.test_split * dataset_length))
            data['training_ids'] = ids[split:]
            data['testing_ids'] = ids[:split]

            with open(file_path, 'wb') as file:
                pickle.dump(data, file)

        return data['training_ids'], data['testing_ids']