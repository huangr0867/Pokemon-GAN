from easydict import EasyDict
import tensorflow as tf
import torch

config = EasyDict()

# MAIN CONFIG
config.MAIN = EasyDict()
config.MAIN.DATA_PATH = 'data/data_ready/'
config.MAIN.NGPU = 1
config.MAIN.NZ = 100
config.MAIN.IMAGE_SIZE = (64, 64)
config.MAIN.SAVE_FREQ = 25  # Checkpoint frequency
config.MAIN.DEVICE = "cuda:0" if torch.cuda.is_available() and config.MAIN.NGPU > 0 else "cpu"
config.MAIN.BATCH_SIZE = 128
config.MAIN.EPOCHS = 1000
config.MAIN.TRANSFORMS = [
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.Resizing(*config.MAIN.IMAGE_SIZE),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    tf.keras.layers.experimental.preprocessing.Normalization(mean=[0.5, 0.5, 0.5], variance=[0.5, 0.5, 0.5])
]

# MODEL CONFIG
config.MODEL = EasyDict()
config.MODEL.DCGAN = EasyDict()
config.MODEL.DCGAN.NC = 3
config.MODEL.DCGAN.NGF = 64
config.MODEL.DCGAN.NDF = 64

config.MODEL.SNGAN = EasyDict()
config.MODEL.SNGAN.NC = 3
config.MODEL.SNGAN.NGF = 64
config.MODEL.SNGAN.NDF = 64

# LOSS CONFIG
config.LOSS = EasyDict()
config.LOSS.BCE = EasyDict()
config.LOSS.BCE.LR = 0.0001
config.LOSS.BCE.BETA1 = 0
config.LOSS.BCE.BETA2 = 0.9

config.LOSS.LB_CE = EasyDict()
config.LOSS.LB_CE.LR = 0.0001
config.LOSS.LB_CE.BETA1 = 0
config.LOSS.LB_CE.BETA2 = 0.9

config.LOSS.WGAN = EasyDict()
config.LOSS.WGAN.LR = 0.00005
config.LOSS.WGAN.BETA1 = 0
config.LOSS.WGAN.BETA2 = 0.9
config.LOSS.WGAN.LAMBDA_GP = 10
config.LOSS.WGAN.CRITICS_ITER = 5
