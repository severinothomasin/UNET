import torch
import os
import numpy as np

import sevtho.data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device Used: ({device}) {torch.cuda.get_device_name(torch.cuda.current_device())}')
print(f'Pytorch Version: {torch.__version__}')

# --------------------------------------
TEST_SPLIT = 0.2
BATCH_SIZE = 4
DATASET_PATH = os.path.join('data','train')
EPOCHS = 100
FILTER_LIST = [16,32,64,128,256]
# --------------------------------------

def get_indices(length, new=False):

    indices = list(range(length))
    np.random.shuffle(indices)
    split = int(np.floor(TEST_SPLIT * len(tumor_dataset)))
    train_indices , test_indices = indices[split:], indices[:split]
    data['train_indices'] = train_indices
    data['test_indices'] = test_indices
    with open(file_path,'wb') as file:
        pickle.dump(data,file)

    return train_indices, test_indices