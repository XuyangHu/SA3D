import tensorflow as tf
import tensorflow.contrib.slim as slim

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