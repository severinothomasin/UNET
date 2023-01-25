from glob import glob as glob

from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import RMSprop

import data
from model import Unet


if __name__ == '__main__':

    dataset_path = "data/train"

    training_images = sorted(glob(f"{dataset_path}/*/image/*.jpg"))
    training_masks = sorted(glob(f"{dataset_path}/*/mask/*.jpg"))

    training_dataset = list(zip(training_images, training_masks))

    batch_size = 8
    image_size = 512
    color_channels = 1

    training_data, validation_data = data.DataLoader(training_dataset)

    input_size = (image_size, image_size, color_channels)

    unet = Unet(input_size)
    unet.summary()

    loss = SparseCategoricalCrossentropy()

    optimizer = RMSprop(learning_rate=0.0001)







