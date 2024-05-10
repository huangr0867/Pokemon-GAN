import tensorflow as tf

def weights_init(m):
    '''
    The function checks the layer's name and initializes its weights with samples from a normal distribution with mean 0 and standard deviation 0.02
    Initializes BatchNormalization with mean 1 and standard deviation 0.02 for weights and sets bias to 0
    '''
    classname = m.__class__.__name__
    if 'Conv' in classname:
        m.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    elif 'BatchNormalization' in classname:
        m.gamma_initializer = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02)
        m.beta_initializer = tf.keras.initializers.Zeros()
