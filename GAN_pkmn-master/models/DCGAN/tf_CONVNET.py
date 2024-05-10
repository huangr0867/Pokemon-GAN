import tensorflow as tf
from config import config

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = tf.keras.Sequential([
            # input is Z, going into a convolution
            tf.keras.layers.Conv2DTranspose(config.MAIN.NZ, (4, 4), strides=(1, 1), padding='valid', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2, inplace=True),
            tf.keras.layers.Dropout(0.5),
            # state size. (ngf*8) x 4 x 4
            tf.keras.layers.Conv2DTranspose(config.MODEL.DCGAN.NGF * 8, (4, 4), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            tf.keras.layers.Conv2DTranspose(config.MODEL.DCGAN.NGF * 4, (4, 4), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            tf.keras.layers.Conv2DTranspose(config.MODEL.DCGAN.NGF * 2, (4, 4), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32
            tf.keras.layers.Conv2DTranspose(config.MODEL.DCGAN.NGF, (4, 4), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2, inplace=True),
            # state size. (nc) x 64 x 64
            tf.keras.layers.Conv2DTranspose(config.MODEL.DCGAN.NC, (4, 4), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.Activation('tanh')
        ])

    def call(self, inputs):
        return self.main(inputs)


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = tf.keras.Sequential([
            # input is (nc) x 64 x 64
            tf.keras.layers.Conv2D(config.MODEL.DCGAN.NC, (4, 4), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            tf.keras.layers.Conv2D(config.MODEL.DCGAN.NDF, (4, 4), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            tf.keras.layers.Conv2D(config.MODEL.DCGAN.NDF * 2, (4, 4), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            tf.keras.layers.Conv2D(config.MODEL.DCGAN.NDF * 4, (4, 4), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            tf.keras.layers.Conv2D(config.MODEL.DCGAN.NDF * 8, (4, 4), strides=(1, 1), padding='valid', use_bias=False),
            tf.keras.layers.LeakyReLU(0.2, inplace=True),
            tf.keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding='valid', use_bias=False)
        ])

    def call(self, inputs):
        return self.main(inputs)