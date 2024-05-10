import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import Progbar

from config import config
from datasets.pkmn_ds import POKEMON_DS
from models import DCGAN
from loss import BCE, LB_CE, WGAN
import argparse
import importlib

# Set up TensorBoard
log_dir = "logs/"
summary_writer = tf.summary.create_file_writer(log_dir)

def train(model, backbone, loss, run_note):
    # Loading Generator and Discriminator
    arch = importlib.import_module(f"models.{model}.{backbone}")
    Generator = arch.Generator().to(config.MAIN.DEVICE)
    Discriminator = arch.Discriminator().to(config.MAIN.DEVICE)

    #Creating the dataset
    image_path = []
    for filename in os.listdir(config.MAIN.DATA_PATH):
        if filename.endswith('.JPG'):
            image_path.append('data/data_ready/' + filename)
    # for filename in glob.glob(os.path.join(config.MAIN.DATA_PATH, "*.jpg")):
    #     image_path.append(filename)

    pkmn_dataset = POKEMON_DS(
        image_path = image_path,
        resize = None,
        transforms = config.MAIN.TRANSFORMS
    )

    # Define loss function and optimizer
    if loss == "BCE":
        loss_fct = BCE()
    elif loss == "LB-CE":
        loss_fct = LB_CE()
    elif loss == "WGAN":
        loss_fct = WGAN()

    optimizerG = Adam(learning_rate=config.LOSS[loss].LR, beta_1=config.LOSS[loss].BETA1, beta_2=config.LOSS[loss].BETA2)
    optimizerD = Adam(learning_rate=config.LOSS[loss].LR, beta_1=config.LOSS[loss].BETA1, beta_2=config.LOSS[loss].BETA2)

    # Training loop
    print(f"Training start for {config.MAIN.EPOCHS} epochs")
    for epoch in range(config.MAIN.EPOCHS):
        print(f"Starting epoch number: {epoch}")

        # Initialize progress bar
        progbar = Progbar(len(pkmn_dataset))

        # Iterate over batches
        for i, batch in enumerate(pkmn_dataset):
            # Training step for discriminator
            with tf.GradientTape() as tape:
                fake_images = Generator(tf.random.normal(shape=(config.MAIN.BATCH_SIZE, config.MAIN.NZ)))
                real_output = Discriminator(batch, training=True)
                fake_output = Discriminator(fake_images, training=True)

                errD = loss_fct.dis_loss(fake_output, real_output)

            gradsD = tape.gradient(errD, Discriminator.trainable_variables)
            optimizerD.apply_gradients(zip(gradsD, Discriminator.trainable_variables))

            # Training step for generator
            with tf.GradientTape() as tape:
                fake_images = Generator(tf.random.normal(shape=(config.MAIN.BATCH_SIZE, config.MAIN.NZ)))
                fake_output = Discriminator(fake_images, training=False)

                errG = loss_fct.gen_loss(fake_output)

            gradsG = tape.gradient(errG, Generator.trainable_variables)
            optimizerG.apply_gradients(zip(gradsG, Generator.trainable_variables))

            # Update progress bar
            progbar.update(i+1)

        # Save checkpoints and log results to TensorBoard
        if epoch % config.MAIN.SAVE_FREQ == 0 or epoch == config.MAIN.EPOCHS-1:
            Generator.save_weights(f"checkpoint/checkpointG-{epoch}.h5")
            Discriminator.save_weights(f"checkpoint/checkpointD-{epoch}.h5")

            with summary_writer.as_default():
                tf.summary.scalar("DCGAN Generator Loss", errG, step=epoch)
                tf.summary.scalar("DCGAN Discriminator Loss", errD, step=epoch)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="DCGAN - SNGAN", default="DCGAN")
parser.add_argument("--backbone", type=str, help="CONVNET - RESNET", default="CONVNET")
parser.add_argument("--loss", type=str, help="BCE - LS-CE - WGAN", default="BCE")
parser.add_argument("--run_note", type=str, help="NOTE OF RUN", default="test")
args = parser.parse_args()

if __name__ == "__main__":
    train(
        model=args.model,
        backbone=args.backbone,
        loss=args.loss,
        run_note=args.run_note
    )
