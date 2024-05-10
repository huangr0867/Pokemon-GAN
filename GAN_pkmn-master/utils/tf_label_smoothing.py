import tensorflow as tf

def label_smoothing(y, real=True):
    '''
    Smoothing the label values
    1 will be between 0.7 and 1.2
    0 will be between 0 and 0.2
    '''
    if real:
        return y - 0.3 + tf.random.uniform(tf.shape(y), minval=0, maxval=0.5)
    else: 
        return y + tf.random.uniform(tf.shape(y), minval=0, maxval=0.2)
