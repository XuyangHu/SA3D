import tensorflow as tf
import tensorflow.contrib.slim as slim


def batch_norm(x, is_training, epsilon=1e-5, decay=0.9, scope="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=decay, updates_collections=None, epsilon=epsilon, scale=True, is_training=is_training, scope=scope)

def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None,
                            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),weights_regularizer=slim.l2_regularizer(0.0005),
                            biases_initializer=None)


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),weights_regularizer=slim.l2_regularizer(0.0005),
                                    biases_initializer=None)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x, name=name)

def relu(x):
    return tf.nn.relu(x)

def l2_criterion(logits ,labels):
    return tf.nn.l2_loss(logits-labels, name='l2_loss')

def kl_criterion(z_mu, z_logvar):
    return tf.reduce_sum(0.5 * (tf.square(z_mu) + tf.square(z_logvar) - tf.log(tf.square(z_logvar) + 1e-8) - 1.0), name='kl_loss')

def vae_criterion(logits, labels, z_mu, z_logvar):
    reg_loss = tf.add_n(slim.losses.get_regularization_losses())
    rec_loss = l2_criterion(logits, labels)
    kl_loss = kl_criterion(z_mu, z_logvar)
    vae_loss = reg_loss + rec_loss + kl_loss
    return reg_loss, rec_loss, kl_loss, vae_loss