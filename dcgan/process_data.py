from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from matplotlib import pyplot as plt

from config import DATA_DIR
# Prepare fashion data.
def fashion_data(visualize=False):
    os.environ['KAGGLE_USERNAME']="ryanhuang888" 
    os.environ['KAGGLE_KEY']="15364762"
    zalando_data_dir = "/Users/ryanhuang/Desktop/cs1430/Pokemon-GAN/dcgan/data/pokemon"
    train_images = tf.keras.utils.image_dataset_from_directory(
    zalando_data_dir, label_mode=None, image_size=(64, 64), batch_size=32
    )
    
    if visualize:
        image_batch = next(iter(train_images))
        random_index = np.random.choice(image_batch.shape[0])
        random_image = image_batch[random_index].numpy().astype("int32")
        plt.axis("off")
        plt.imshow(random_image)
        plt.show()
    
    # Normalize the images to [-1, 1] which is the range of the tanh activation
    train_images = train_images.map(lambda x: (x - 127.5) / 127.5)
    return train_images
    
def input_data(path, batch_size):
    # Get list of image file paths
    image_paths = [os.path.join(path, image) for image in os.listdir(path) if image.endswith('.JPG')]

    
    # Define a function to read and preprocess each image
    def preprocess_image(image_path):
    # Open image using PIL
        with Image.open(image_path) as img:
            # Convert image to RGB if it's not in that mode
            img = img.convert('RGB')
            # Resize image to desired shape (e.g., 28x28 for MNIST)
            img = img.resize((28, 28))
            # Convert image to numpy array
            image = np.array(img)
            
            # Normalize pixel values to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            return image
        
    print(image_paths)
    images = []
    for image in image_paths:
        images.append(preprocess_image(image))
    
    # Create a dataset from the list of file paths
    dataset = tf.data.Dataset.from_tensor_slices(images)
    
    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=len(image_paths)).batch(batch_size)
    
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    