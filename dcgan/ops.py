import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from tensorflow.keras import initializers

def image_summary(*args, **kwargs):
    return tf.summary.image(*args, **kwargs)

def scalar_summary(*args, **kwargs):
    return tf.summary.scalar(*args, **kwargs)

def histogram_summary(*args, **kwargs):
    return tf.summary.histogram(*args, **kwargs)

def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

class batch_norm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm", **kwargs):
        super(batch_norm, self).__init__(name=name, **kwargs)
        self.epsilon = epsilon
        self.momentum = momentum

    def call(self, x, training=True):
        return tf.keras.layers.BatchNormalization(
            momentum=self.momentum,
            epsilon=self.epsilon,
            trainable=training,
            name=self.name
        )(x, training=training)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.shape
    y_shapes = y.shape
    return concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    return Conv2D(
        filters=output_dim,
        kernel_size=(k_h, k_w),
        strides=(d_h, d_w),
        padding='same',
        kernel_initializer=initializers.RandomNormal(stddev=stddev),
        bias_initializer='zeros',
        name=name
    )(input_)

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
    return Conv2DTranspose(
        filters=output_shape[-1],
        kernel_size=(k_h, k_w),
        strides=(d_h, d_w),
        padding='same',
        kernel_initializer=initializers.RandomNormal(stddev=stddev),
        bias_initializer='zeros',
        name=name
    )(input_)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    with tf.name_scope(scope or "Linear"):
        matrix = tf.Variable(tf.random.truncated_normal([input_.shape[1], output_size], stddev=stddev), name="Matrix")
        bias = tf.Variable(tf.constant(bias_start, shape=[output_size]), name="bias")
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
