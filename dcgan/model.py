import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np

from ops import *
from utils import *

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class DCGAN(tf.keras.Model):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
                 batch_size=64, sample_num=64, output_height=64, output_width=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
        """
        Args:
            input_height: Height of input images.
            input_width: Width of input images.
            crop: Whether to crop the input images.
            batch_size: Batch size for training.
            sample_num: Number of samples to generate.
            output_height: Height of output images.
            output_width: Width of output images.
            y_dim: Dimension of y.
            z_dim: Dimension of z.
            gf_dim: Dimension of generator filters in the first conv layer.
            df_dim: Dimension of discriminator filters in the first conv layer.
            gfc_dim: Dimension of generator units for the fully connected layer.
            dfc_dim: Dimension of discriminator units for the fully connected layer.
            c_dim: Dimension of image color.
            dataset_name: Name of the dataset.
            input_fname_pattern: File pattern for input images.
            checkpoint_dir: Directory to save checkpoints.
            sample_dir: Directory to save samples.
        """
        super(DCGAN, self).__init__()
        self.sess = sess
        self.crop = crop
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.c_dim = c_dim
        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir

        self.data = [f"./data/pokemon/{file}" for file in os.listdir("./data/pokemon/")]
        imreadImg = imread(self.data[0])
        if len(imreadImg.shape) >= 3:
            self.c_dim = imread(self.data[0]).shape[-1]
        else:
            self.c_dim = 1
        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_model(self):
        self.y = None

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.keras.Input(shape=image_dims, name='real_images')

        inputs = self.inputs

        self.z = tf.keras.Input(shape=(self.z_dim,), name='z')

        self.G = self.generator(self.z, self.y)
        self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
        self.sampler = self.sampler(self.z, self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

        self.d_bn1 = tf.keras.layers.BatchNormalization(name='d_bn1')
        self.d_bn2 = tf.keras.layers.BatchNormalization(name='d_bn2')
        self.d_bn3 = tf.keras.layers.BatchNormalization(name='d_bn3')
        self.g_bn0 = tf.keras.layers.BatchNormalization(name='g_bn0')
        self.g_bn1 = tf.keras.layers.BatchNormalization(name='g_bn1')
        self.g_bn2 = tf.keras.layers.BatchNormalization(name='g_bn2')
        self.g_bn3 = tf.keras.layers.BatchNormalization(name='g_bn3')

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.optimizer = tf.keras.optimizers.Adam()

    def train(self, config):
        # Training loop
        for epoch in range(config.epoch):
            # Loop over batches
            for idx in range(0, len(self.data), config.batch_size):
                batch_files = self.data[idx:idx + config.batch_size]
                batch = [
                    get_image(batch_file,
                              input_height=self.input_height,
                              input_width=self.input_width,
                              resize_height=self.output_height,
                              resize_width=self.output_width,
                              crop=self.crop,
                              grayscale=self.grayscale) for batch_file in batch_files]
                if self.grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.normal(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # Training step for discriminator
                with tf.GradientTape() as tape:
                    d_loss_real = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                                labels=tf.ones_like(self.D)))
                    d_loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                                labels=tf.zeros_like(self.D_)))
                    d_loss = d_loss_real + d_loss_fake

                gradients = tape.gradient(d_loss, self.d_vars)
                self.optimizer.apply_gradients(zip(gradients, self.d_vars))

                # Training step for generator
                with tf.GradientTape() as tape:
                    g_loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                                labels=tf.ones_like(self.D_)))

                gradients = tape.gradient(g_loss, self.g_vars)
                self.optimizer.apply_gradients(zip(gradients, self.g_vars))

                # Update generator twice to make sure that d_loss does not go to zero
                with tf.GradientTape() as tape:
                    g_loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                                labels=tf.ones_like(self.D_)))

                gradients = tape.gradient(g_loss, self.g_vars)
                self.optimizer.apply_gradients(zip(gradients, self.g_vars))

                # Print training progress
                print(f"Epoch: {epoch}, Batch: {idx}/{len(self.data)}, d_loss: {d_loss.numpy()}, g_loss: {g_loss.numpy()}")

            # Save model checkpoints
            if (epoch + 1) % 10 == 0:
                self.save(config.checkpoint_dir, epoch)

    def discriminator(self, image, y=None, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4

    def generator(self, z, y=None):
      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # Ensure that linear function returns a TensorFlow tensor
      z_linear = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin')

      h0 = tf.reshape(
          z_linear,
          [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = lrelu(self.g_bn0(h0))
      
      h1 = lrelu(self.g_bn1(deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')))
      h2 = lrelu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')))
      h3 = lrelu(self.g_bn3(deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')))
      h4 = tf.nn.tanh(deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4'))
      return h4



    def sampler(self, z, y=None):
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        h0 = tf.reshape(
            linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = lrelu(self.g_bn0(h0, train=False))
        h1 = lrelu(self.g_bn1(deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')))
        h2 = lrelu(self.g_bn2(deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')))
        h3 = lrelu(self.g_bn3(deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')))
        h4 = tf.nn.tanh(deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4'))
        return h4

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.save_weights(os.path.join(checkpoint_dir, model_name))

    def load(self, checkpoint_dir):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        self.load_weights(os.path.join(checkpoint_dir, model_name))
