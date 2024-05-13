import tensorflow as tf
NUM_EPOCHS = 1000
D_LR = 0.0001
G_LR = 0.0003

LATENT_DIM = 100 
WEIGHT_INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02) 
CHANNELS = 3 # for colored img

DATA_DIR = '../data'