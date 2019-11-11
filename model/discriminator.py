import tensorflow as tf
from model.ops import *


class Discriminator:
    def __init__(self, opt, name, use_sigmoid=False):
        self.name = name
        self.is_training = opt.is_training
        self.norm = opt.norm
        self.reuse = False
        self.ndf = opt.ndf
        self.use_sigmoid = use_sigmoid

    def __call__(self, input):
        with tf.variable_scope(self.name):
            C64 = Ck(input, self.ndf, reuse=self.reuse, norm=None, is_training=self.is_training, name='C64')
            C128 = Ck(C64, self.ndf * 2, reuse=self.reuse, norm=self.norm, is_training=self.is_training, name='C128')
            C256 = Ck(C128, self.ndf * 4, reuse=self.reuse, norm=self.norm, is_training=self.is_training, name='C256')
            C512 = Ck(C256, self.ndf * 8, stride=1, reuse=self.reuse, norm=self.norm, is_training=self.is_training,
                      name='C512')
            output = last_conv(C512, reuse=self.reuse, use_sigmoid=self.use_sigmoid, name='output')

            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output
