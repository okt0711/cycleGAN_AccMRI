import tensorflow as tf


def mae_criterion(pred, target):
    return tf.reduce_mean(tf.abs(pred - target))


def mse_criterion(pred, target):
    return tf.reduce_mean(tf.squared_difference(pred, target))


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def gradient_penalty(fake, real, batch_size, discriminator):
    fY = tf.shape(fake)[1]
    rY = tf.shape(real)[1]
    padY = tf.abs(fY - rY)

    real = tf.cond(tf.greater(fY, rY), lambda: tf.pad(real, [[0, 0], [padY / 2, padY / 2], [0, 0], [0, 0]], 'CONSTANT'),
                   lambda: real)
    fake = tf.cond(tf.greater(rY, fY), lambda: tf.pad(fake, [[0, 0], [padY / 2, padY / 2], [0, 0], [0, 0]], 'CONSTANT'),
                   lambda: fake)

    alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
    interpolates = real + alpha * (fake - real)
    gradients = tf.gradients(discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

    return gradient_penalty
