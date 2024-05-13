from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from matplotlib import pyplot as plt

from config import DATA_DIR

# prepare pokemon data
def process_data(data_dir, visualize=False):
    """
    Process image data from the specified directory.

    Args:
        data_dir (str): Path to the directory containing the image data.
        visualize (bool, optional): Whether to visualize a random image from the dataset. Defaults to False.

    Returns:
        tf.data.Dataset: Processed image dataset.
    """
    data = data_dir
    train_images = tf.keras.utils.image_dataset_from_directory(
        data, label_mode=None, image_size=(64, 64), batch_size=32
    )
    
    if visualize:
        image_batch = next(iter(train_images))
        random_index = np.random.choice(image_batch.shape[0])
        random_image = image_batch[random_index].numpy().astype("int32")
        plt.axis("off")
        plt.imshow(random_image)
        plt.show()
    
    train_images = train_images.map(lambda x: (x - 127.5) / 127.5)
    return train_images