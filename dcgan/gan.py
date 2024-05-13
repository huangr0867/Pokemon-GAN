import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from matplotlib import pyplot as plt

class DCGAN(keras.Model):
    """
    Deep Convolutional Generative Adversarial Network (DCGAN) class.

    This class combines a discriminator and a generator to form a complete GAN model.

    Args:
        discriminator (keras.Model): The discriminator model.
        generator (keras.Model): The generator model.
        latent_dim (int): Dimensionality of the latent space.

    Attributes:
        discriminator (keras.Model): The discriminator model.
        generator (keras.Model): The generator model.
        latent_dim (int): Dimensionality of the latent space.
        d_loss_metric (keras.metrics.Mean): Metric to track discriminator loss.
        g_loss_metric (keras.metrics.Mean): Metric to track generator loss.
        d_optimizer (tf.keras.optimizers.Optimizer): Discriminator optimizer.
        g_optimizer (tf.keras.optimizers.Optimizer): Generator optimizer.
        loss_fn (function): Loss function for the GAN.
    """
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        """
        Compile the GAN model.

        Args:
            d_optimizer (tf.keras.optimizers.Optimizer): Discriminator optimizer.
            g_optimizer (tf.keras.optimizers.Optimizer): Generator optimizer.
            loss_fn (function): Loss function for the GAN.
        """
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images, training=True):
        """
        Perform one training step.

        Args:
            real_images (tensor): Batch of real images.
            training (bool): Whether the model is training or not.

        Returns:
            dict: Dictionary containing discriminator and generator losses.
        """
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            pred_real = self.discriminator(real_images, training)
            real_labels = tf.ones((batch_size, 1))
            real_labels += 0.05 * tf.random.uniform(tf.shape(real_labels)) 
            d_loss_real = self.loss_fn(real_labels, pred_real)

            fake_images = self.generator(noise)
            pred_fake = self.discriminator(fake_images, training)
            fake_labels = tf.zeros((batch_size, 1))
            d_loss_fake = self.loss_fn(fake_labels, pred_fake)

            d_loss = (d_loss_real + d_loss_fake)/2
        gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training)
            pred_fake = self.discriminator(fake_images, training)
            g_loss = self.loss_fn(misleading_labels, pred_fake)
        # gradients
        gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        # weights
        self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}
    

class GANMonitor(keras.callbacks.Callback):
    """
    Callback to generate and save images during training.

    Args:
        num_img (int): Number of images to generate and save.
        latent_dim (int): Dimensionality of the latent space.
    """
    def __init__(self, num_img=3, latent_dim=100):
        self.num_img = num_img
        self.latent_dim = latent_dim

        # random noise seed
        self.seed = tf.random.normal([16, latent_dim])

    def on_epoch_end(self, epoch, logs=None):
        """
        Generate and save images at the end of each epoch.

        Args:
            epoch (int): Current epoch number.
            logs (dict): Dictionary of logs.
        """
        generated = self.model.generator(self.seed)
        generated = (generated * 127.5) + 127.5
        generated.numpy()

        fig = plt.figure(figsize=(4, 4))
        for i in range(self.num_img):
            plt.subplot(4, 4, i+1)
            img = keras.utils.array_to_img(generated[i]) 
            plt.imshow(img)
            plt.axis('off')
        plt.savefig('./results/epoch_{:03d}.png'.format(epoch))

    def on_train_end(self, logs=None):
        """
        Callback at the end of training.

        Args:
            logs (dict): Dictionary of logs.
        """
        self.model.generator.save('./trained_generator/generator.h5')