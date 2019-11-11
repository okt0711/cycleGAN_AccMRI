import tensorflow as tf
from model.ops import *
from utils import *


class UnetGenerator:
    def __init__(self, opt, name, res=False):
        self.name = name
        self.reuse = False
        self.ngf = opt.ngf
        self.norm = opt.norm
        self.is_training = opt.is_training
        self.nC = opt.nC
        self.res = res

    def __call__(self, input):
        with tf.variable_scope(self.name):
            sz = tf.shape(input)
            padY = (16 - sz[1] % 16)
            padded_input = tf.pad(input, [[0, 0], [padY / 2, padY / 2], [0, 0], [0, 0]], 'CONSTANT')

            CBR_64_down1 = CBR_k(padded_input, self.ngf, is_training=self.is_training, norm=None, reuse=self.reuse,
                                 name='CBR_64_down1')
            CBR_64_down2 = CBR_k(CBR_64_down1, self.ngf, is_training=self.is_training, norm=None, reuse=self.reuse,
                                 name='CBR_64_down2')
            Pool_64 = Pool(CBR_64_down2, reuse=self.reuse, name='Pool_64')

            CBR_128_down1 = CBR_k(Pool_64, self.ngf * 2, is_training=self.is_training, norm=self.norm, reuse=self.reuse,
                                  name='CBR_128_down1')
            CBR_128_down2 = CBR_k(CBR_128_down1, self.ngf * 2, is_training=self.is_training, norm=self.norm,
                                  reuse=self.reuse, name='CBR_128_down2')
            Pool_128 = Pool(CBR_128_down2, reuse=self.reuse, name='Pool_128')

            CBR_256_down1 = CBR_k(Pool_128, self.ngf * 4, is_training=self.is_training, norm=self.norm,
                                  reuse=self.reuse, name='CBR_256_down1')
            CBR_256_down2 = CBR_k(CBR_256_down1, self.ngf * 4, is_training=self.is_training, norm=self.norm,
                                  reuse=self.reuse, name='CBR_256_down2')
            Pool_256 = Pool(CBR_256_down2, reuse=self.reuse, name='Pool_256')

            CBR_512_down1 = CBR_k(Pool_256, self.ngf * 8, is_training=self.is_training, norm=self.norm,
                                  reuse=self.reuse, name='CBR_512_down1')
            CBR_512_down2 = CBR_k(CBR_512_down1, self.ngf * 8, is_training=self.is_training, norm=self.norm,
                                  reuse=self.reuse, name='CBR_512_down2')
            Pool_512 = Pool(CBR_512_down2, reuse=self.reuse, name='Pool_512')

            CBR_1024_1 = CBR_k(Pool_512, self.ngf * 16, is_training=self.is_training, norm=self.norm, reuse=self.reuse,
                               name='CBR_1024_1')
            CBR_1024_2 = CBR_k(CBR_1024_1, self.ngf * 16, is_training=self.is_training, norm=self.norm,
                               reuse=self.reuse, name='CBR_1024_2')
            up_512 = up_k(CBR_1024_2, self.ngf * 8, reuse=self.reuse, name='up_512')

            CC_1024 = CC(CBR_512_down2, up_512, reuse=self.reuse, name='CC_1024')
            CBR_512_up1 = CBR_k(CC_1024, self.ngf * 8, is_training=self.is_training, norm=self.norm, reuse=self.reuse,
                                name='CBR_512_up1')
            CBR_512_up2 = CBR_k(CBR_512_up1, self.ngf * 8, is_training=self.is_training, norm=self.norm,
                                reuse=self.reuse, name='CBR_512_up2')
            up_256 = up_k(CBR_512_up2, self.ngf * 4, reuse=self.reuse, name='up_256')

            CC_512 = CC(CBR_256_down2, up_256, reuse=self.reuse, name='CC_512')
            CBR_256_up1 = CBR_k(CC_512, self.ngf * 4, is_training=self.is_training, norm=self.norm, reuse=self.reuse,
                                name='CBR_256_up1')
            CBR_256_up2 = CBR_k(CBR_256_up1, self.ngf * 4, is_training=self.is_training, norm=self.norm,
                                reuse=self.reuse, name='CBR_256_up2')
            up_128 = up_k(CBR_256_up2, self.ngf * 2, reuse=self.reuse, name='up_128')

            CC_256 = CC(CBR_128_down2, up_128, reuse=self.reuse, name='CC_256')
            CBR_128_up1 = CBR_k(CC_256, self.ngf * 2, is_training=self.is_training, norm=self.norm, reuse=self.reuse,
                                name='CBR_128_up1')
            CBR_128_up2 = CBR_k(CBR_128_up1, self.ngf * 2, is_training=self.is_training, norm=self.norm,
                                reuse=self.reuse, name='CBR_128_up2')
            up_64 = up_k(CBR_128_up2, self.ngf, reuse=self.reuse, name='up_64')

            CC_128 = CC(CBR_64_down2, up_64, reuse=self.reuse, name='CC_128')
            CBR_64_up1 = CBR_k(CC_128, self.ngf, is_training=self.is_training, norm=self.norm, reuse=self.reuse,
                               name='CBR_64_up1')
            CBR_64_up2 = CBR_k(CBR_64_up1, self.ngf, is_training=self.is_training, norm=self.norm, reuse=self.reuse,
                               name='CBR_64_up2')
            C1x1 = Conv1x1(CBR_64_up2, self.nC * 2, reuse=self.reuse, name='Conv1x1')

            if self.res:
                output = tf.slice(C1x1 + padded_input, [0, padY, 0, 0], sz, name='output')
            else:
                output = tf.slice(C1x1, [0, padY, 0, 0], sz, name='output')

            self.reuse = True
            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

            return output


class FFTGenerator:
    def __init__(self, opt, name):
        self.name = name
        self.nC = opt.nC

    def __call__(self, input, mask):
        with tf.variable_scope(self.name):
            full_k = myTFfft2(tf_ri2comp(input))
            down_k = tf.multiply(full_k, tf.cast(mask, tf.complex64))
            output = tf_comp2ri(myTFifft2(down_k))

            return output
