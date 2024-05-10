import os
import inspect
import tensorflow as tf

def save_checkpoint(generator, discriminator, optiG, optiD, epoch, loss_fct, errG, errD, run_note):
    '''
    Save G & D in the checkpoint repository
    '''
    checkpoint_dir = "checkpoint/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    loss_file = loss_fct.__class__.__name__
    G_file = generator.__class__.__name__
    G_folder = os.path.basename(os.path.dirname(inspect.getfile(generator)))
    D_file = discriminator.__class__.__name__
    D_folder = os.path.basename(os.path.dirname(inspect.getfile(discriminator)))

    tf.saved_model.save(generator, os.path.join(checkpoint_dir, f"checkpointG-{G_folder}_{G_file}_{loss_file}_{epoch}_{round(errG.numpy(), 2)}_{run_note}"))
    tf.saved_model.save(discriminator, os.path.join(checkpoint_dir, f"checkpointD-{D_folder}_{D_file}_{loss_file}_{epoch}_{round(errD.numpy(), 2)}_{run_note}"))
