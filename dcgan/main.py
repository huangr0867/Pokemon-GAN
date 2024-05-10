import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from matplotlib import pyplot as plt
import numpy as np

import os
import argparse
from process_data import *
from config import *
from model import *
from dcgan import *

def main():
    parser = argparse.ArgumentParser(description="Training DCGAN on Pokemons")
    # MNIST Dataset parameters.
    parser.add_argument("--lr_generator", type=float, default=0.0002)
    parser.add_argument("--lr_discriminator", type=float, default=0.0002)
    parser.add_argument("--training_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--display_step", type=int, default=5)
    
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--dataset", type=str, default='fashion')
    parser.add_argument("--test", type=bool, default=False)
    
    args = vars(parser.parse_args())
    lr_generator = args['lr_generator']
    lr_discriminator = args['lr_discriminator']
    training_steps = args['training_steps']
    batch_size = args['batch_size']
    display_step = args['display_step']
    noise_dim = args['noise_dim']
    dataset = args['dataset']
    test = args['test']
    
    
    if dataset == 'fashion':
        train_images = fashion_data()
        # build the generator model
        generator = build_generator()
        generator.summary()
        # build the discriminator model
        discriminator = build_discriminator(64, 64, 3)  
        discriminator.summary()
        dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)
        D_LR = 0.0001 # UPDATED: discriminator learning rate
        G_LR = 0.0003 # UPDATED: generator learning rate

        dcgan.compile(
            d_optimizer=keras.optimizers.Adam(learning_rate=D_LR, beta_1 = 0.5),
            g_optimizer=keras.optimizers.Adam(learning_rate=G_LR, beta_1 = 0.5),  
            loss_fn=keras.losses.BinaryCrossentropy(),
        )
        NUM_EPOCHS = 50 # number of epochs
        dcgan.fit(train_images, epochs=NUM_EPOCHS, callbacks=[GANMonitor(num_img=16, latent_dim=LATENT_DIM)])
        
        
    else:
        pass


if __name__ == "__main__":
    main()