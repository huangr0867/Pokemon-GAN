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
    
    parser.add_argument("--dataset", type=str, default=DATA_DIR)
    parser.add_argument("--visualize", type=bool, default=False)
    
    args = vars(parser.parse_args())
    dataset = args['dataset']
    visualize = args['visualize']

    train_images = process_data(dataset, visualize)
    
    # generator
    generator = build_generator()
    generator.summary()
    
    # discriminator
    discriminator = build_discriminator(64, 64, 3)  
    discriminator.summary()
    dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)

    dcgan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=D_LR, beta_1=0.5),
        g_optimizer=keras.optimizers.Adam(learning_rate=G_LR, beta_1=0.5),  
        loss_fn=keras.losses.BinaryCrossentropy(),
    )
    
    dcgan.fit(train_images, epochs=NUM_EPOCHS, callbacks=[GANMonitor(num_img=16, latent_dim=LATENT_DIM)])
    

if __name__ == "__main__":
    main()