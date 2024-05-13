from tensorflow.keras import Model, layers
from config import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ReLU, Reshape, Conv2DTranspose, Conv2D, BatchNormalization, LeakyReLU, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_generator():
    """
    Generator model for DCGAN.

    Returns:
        keras.Model: Generator model.
    """
    model = Sequential(name='generator')

    model.add(Dense(8 * 8 * 512, input_dim=LATENT_DIM))
    model.add(ReLU())

    # reshape
    model.add(Reshape((8, 8, 512)))

    # conv2dT + BN + ReLU
    model.add(Conv2DTranspose(256, (4, 4), strides=(2, 2),padding="same", kernel_initializer=WEIGHT_INIT))
    model.add((ReLU()))

    # conv2dT + BN + ReLU
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2),padding="same", kernel_initializer=WEIGHT_INIT))
    model.add((ReLU()))

    # conv2dT + BN + ReLU
    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2),padding="same", kernel_initializer=WEIGHT_INIT))
    model.add((ReLU()))

    # conv2d + tanh
    model.add(Conv2D(CHANNELS, (4, 4), padding="same", activation="tanh"))

    return model

def build_discriminator(height, width, depth, alpha=0.2):
    """
    Discriminator model for DCGAN.

    Args:
        height (int): Height of input images.
        width (int): Width of input images.
        depth (int): Depth (channels) of input images.
        alpha (float): Slope of the leaky ReLU activation function.

    Returns:
        keras.Model: Discriminator model.
    """
    model = Sequential(name='discriminator')
    input_shape = (height, width, depth)

    # conv2d + BN + leaky ReLU
    model.add(Conv2D(64, (4, 4), padding="same", strides=(2, 2),
        input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))

    # conv2d + BN + leaky ReLU
    model.add(Conv2D(128, (4, 4), padding="same", strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))

    # conv2d + BN + leaky ReLU
    model.add(Conv2D(128, (4, 4), padding="same", strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=alpha))

    # flatten + dropout
    model.add(Flatten())
    model.add(Dropout(0.3))

    model.add(Dense(1, activation="sigmoid"))

    return model

