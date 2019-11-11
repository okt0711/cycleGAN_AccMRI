import tensorflow as tf
import numpy as np
import random

dtype = tf.float32


def myNumExt(s):
    head = s.rstrip('0123456789')
    tail = s[len(head):]
    return int(tail)


def myTFfft2(inp):
    return tf.transpose(tf.fft2d(tf.transpose(tf.cast(inp, tf.complex64), [0, 3, 1, 2])), [0, 2, 3, 1])


def myTFifft2(inp):
    return tf.transpose(tf.ifft2d(tf.transpose(inp, [0, 3, 1, 2])), [0, 2, 3, 1])


def tf_imgri2ssos(img_ri):
    hnOut = int(int(img_ri.shape[3]) / 2)
    i_real = tf.slice(img_ri, [0, 0, 0, 0], [-1, -1, -1, hnOut])
    i_imag = tf.slice(img_ri, [0, 0, 0, hnOut], [-1, -1, -1, hnOut])
    i_ssos = tf.sqrt(tf.reduce_sum(tf.square(i_real) + tf.square(i_imag), axis=3, keep_dims=True))
    i_ssos = tf.cast(i_ssos * 255.0 / tf.reduce_max(i_ssos), dtype=tf.uint8)
    return i_ssos


def tf_ri2comp(ri):
    hnOut = int(int(ri.shape[3]) / 2)
    real = tf.slice(ri, [0, 0, 0, 0], [-1, -1, -1, hnOut])
    imag = tf.slice(ri, [0, 0, 0, hnOut], [-1, -1, -1, hnOut])
    comp = tf.complex(real, imag)
    return comp


def tf_comp2ri(comp):
    return tf.concat([tf.cast(tf.real(comp), dtype), tf.cast(tf.imag(comp), dtype)], axis=3)


class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def __call__(self, image):
        if self.pool_size <= 0:
            return image

        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randrange(0, self.pool_size)
                tmp = self.images[random_id].copy()
                self.images[random_id] = image.copy()
                return tmp
            else:
                return image


def make_mask(nY, nX, nC, DSrate):
    mask = np.zeros((nY, nX), dtype=np.float32)
    masks = np.empty((1, nY, nX, nC), dtype=np.float32)
    if DSrate == 4:
        nACS = round(nY * 0.08)
    else:
        nACS = round(nY * 0.04)

    ACS_s = round((nY - nACS) / 2)
    ACS_e = ACS_s + nACS

    max_ = int(nY / DSrate)

    for i in range(max_):
        r = np.random.randint(1, nY)
        mask[r, :] = 1

    mask[ACS_s:ACS_e, :] = 1
    mask = np.fft.fftshift(mask)
    masks[0, :, :, :] = np.tile(mask[:, :, np.newaxis], [1, 1, nC])
    return masks
