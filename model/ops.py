import tensorflow as tf
import tensorflow.contrib.layers as li

dtype = tf.float32


# Generator layers

def CBR_k(input, k, reuse=False, norm='instance', is_training=True, name='CBR_k'):
    """
    3x3 convolution-normalization-activation layer with k filters and stride 1
    :param input: 4D tensor
    :param k: the number of filters, int
    :param reuse: bool
    :param norm: the type of normalization, 'instance' or 'batch' or None
    :param is_training: bool
    :param name: string
    :return: 4D tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        conv = tf.layers.conv2d(input, filters=k, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                kernel_initializer=li.xavier_initializer())
        normalized = _norm(conv, is_training, norm)
        output = tf.nn.leaky_relu(normalized)
        return output


def Pool(input, reuse=False, name='Pool'):
    """
    2x2 max-pooling layer with stride 2
    :param input: 4D tensor
    :param reuse: bool
    :param name: string
    :return: 4D tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        output = tf.layers.max_pooling2d(input, pool_size=(2, 2), strides=(2, 2), padding='valid')
        return output


def up_k(input, k, reuse=False, name='up_k'):
    """
    2x2 transpose convolution layer with stride 2
    :param input: 4D tensor
    :param k: the number of filters, int
    :param reuse: bool
    :param name: string
    :return: 4D tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        output = tf.layers.conv2d_transpose(input, filters=k, kernel_size=(2, 2), strides=(2, 2),
                                            kernel_initializer=li.xavier_initializer())
        return output


def CC(down, up, reuse=False, name='CC'):
    """
    concatenate two inputs
    :param down: 4D tensor
    :param up: 4D tensor
    :param reuse: bool
    :param name: string
    :return: 4D tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        output = tf.concat([down, up], axis=3)
        return output


def Conv1x1(input, k, reuse=False, name='Conv1x1'):
    """
    1x1 convolution layer with stride 1
    :param input: 4D tensor
    :param k: the number of filters, int
    :param reuse: bool
    :param name: string
    :return: 4D tensor
    """
    with tf.variable_scope(name, reuse=reuse):
        output = tf.layers.conv2d(input, filters=k, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                  kernel_initializer=li.xavier_initializer())
        return output


# Discriminator layers
def Ck(input, k, slope=0.2, stride=2, reuse=False, norm='instance', is_training=True, name='Ck'):
    """
    4x4 convolution-normalization-leakyReLU layer with k filters and stride 2
    :param input: 4D tensor
    :param k: the number of filters, int
    :param slope: LeakyReLU's slope, float
    :param stride: int
    :param reuse: bool
    :param norm: 'the type of normalization, 'instance' or 'batch' or None
    :param is_training: bool
    :param name: string
    :return: 4D tensor
    """

    with tf.variable_scope(name, reuse=reuse):
        conv = tf.layers.conv2d(input, filters=k, kernel_size=(4, 4), strides=(stride, stride), padding='same',
                                kernel_initializer=li.xavier_initializer())
        normalized = _norm(conv, is_training, norm)
        output = tf.nn.leaky_relu(normalized, alpha=slope)
        return output


def last_conv(input, reuse=False, use_sigmoid=False, name=None):
    """
    last convolution layer of discriminator network (1 filter with size 4x4, stride 1)
    :param input: 4D tensor
    :param reuse: bool
    :param use_sigmoid: bool (False if use lsgan or wgan)
    :param name: string
    :return: 4D tensor
    """

    with tf.variable_scope(name, reuse=reuse):
        output = tf.layers.conv2d(input, filters=1, kernel_size=(4, 4), strides=(1, 1), padding='same',
                                  kernel_initializer=li.xavier_initializer())
        if use_sigmoid:
            output = tf.sigmoid(output)
        return output


# Helpers
def _instance_norm(input):
    with tf.variable_scope('instance_norm'):
        return li.instance_norm(input, center=True, scale=True, epsilon=1e-5)


def _batch_norm(input, is_training):
    with tf.variable_scope('batch_norm'):
        return li.batch_norm(input, decay=0.9, scale=True, epsilon=1e-5, updates_collections=None,
                             is_training=is_training)


def _norm(input, is_training, norm='instance'):
    if norm == 'instance':
        return _instance_norm(input)
    elif norm == 'batch':
        return _batch_norm(input, is_training)
    else:
        return input

