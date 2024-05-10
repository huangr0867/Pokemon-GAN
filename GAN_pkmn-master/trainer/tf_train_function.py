import tensorflow as tf
from config import config

class Trainer:
    def __init__(self, generator, discriminator, optiD, optiG, loss, device):
        self.G = generator
        self.D = discriminator
        self.optiD = optiD
        self.optiG = optiG
        self.loss = loss
        self.device = device

    def dis_step(self, data):
        # REAL DATA
        real_images = data["images"]
        real_images = tf.convert_to_tensor(real_images, dtype=tf.float32)
        real_images = tf.image.resize(real_images, (64, 64))  # Resize if necessary
        real_images = tf.transpose(real_images, perm=[0, 3, 1, 2])  # NHWC to NCHW
        real_images = tf.cast(real_images, dtype=tf.float32)
        # FAKE DATA
        noise = tf.random.normal((config.MAIN.BATCH_SIZE, config.MAIN.NZ, 1, 1), dtype=tf.float32)
        # LABELS
        label_real = tf.ones((config.MAIN.BATCH_SIZE,), dtype=tf.float32)
        label_fake = tf.zeros((config.MAIN.BATCH_SIZE,), dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # D OUTPUT ON REAL DATA
            output_real = self.D(real_images, training=True)
            output_real = tf.reshape(output_real, (-1,))
            # D OUTPUT ON FAKE DATA
            fake = self.G(noise, training=False)
            output_fake = self.D(fake, training=True)
            output_fake = tf.reshape(output_fake, (-1,))
            # DIS LOSS
            errD = self.loss.dis_loss(output_fake=output_fake, output_real=output_real, label_real=label_real, label_fake=label_fake)
            # WE ADD GRADIENT PENALTY IF WGAN LOSS
            if self.loss.name == "WGAN_loss":
                errD_GP = self.loss.compute_gradient_penalty_loss(real_images=real_images, fake_images=fake, discriminator=self.D, gp_scale=config.LOSS.WGAN.LAMBDA_GP)
                errD += errD_GP

        # GRADIENT BACKPROP & OPTIMIZER STEP
        gradients = tape.gradient(errD, self.D.trainable_variables)
        self.optiD.apply_gradients(zip(gradients, self.D.trainable_variables))

        # COMPUTING PROBS
        D_x, D_G_z1 = self.loss.compute_probs(output_real=output_real, output_fake=output_fake)
        return errD, D_x, D_G_z1, fake

    def gen_step(self, fake_images):
        # LABEL FOR GENERATOR
        label_gen = tf.ones((config.MAIN.BATCH_SIZE,), dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # OUTPUT OF DISCRIMINATOR AFTER TRAINING STEP
            output_gen = self.D(fake_images, training=False)
            output_gen = tf.reshape(output_gen, (-1,))
            # LOSS WITH LABEL AS REAL
            errG = self.loss.gen_loss(output_gen=output_gen, label_gen=label_gen)

        gradients = tape.gradient(errG, self.G.trainable_variables)
        self.optiG.apply_gradients(zip(gradients, self.G.trainable_variables))

        # COMPUTING PROBS 
        D_G_z2 = tf.reduce_mean(tf.sigmoid(output_gen)).numpy()
        return errG, D_G_z2

    def training_step(self, dataloader):
        D_losses = []
        G_losses = []

        for i, data in enumerate(dataloader):
            ######################
            # DISCRIMINATOR STEP #
            ######################
            errD, D_x, D_G_z1, fake_images = self.dis_step(data=data)

            ##################
            # GENERATOR STEP #
            ##################
            if (self.loss.name == "WGAN_loss") and (i % config.LOSS.WGAN.CRITICS_ITER != 0):
                pass
            else:
                errG, D_G_z2 = self.gen_step(fake_images)

            D_losses.append(errD)
            G_losses.append(errG)

        avg_D_loss = tf.reduce_mean(D_losses).numpy()
        avg_G_loss = tf.reduce_mean(G_losses).numpy()
        print(f"Discriminator Loss = {avg_D_loss}, Prediction on Real data = {D_x} and fake data = {D_G_z1}")
        print(f"Generator Loss = {avg_G_loss}, and fooling power = {D_G_z2}")
        return avg_D_loss, D_x, D_G_z1, avg_G_loss, D_G_z2
