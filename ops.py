import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

class batch_norm(object):
            # h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),decay=0.9,updates_collections=None,epsilon=0.00001,scale=True,scope="d_h1_conv"))
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)

def binary_cross_entropy(preds, targets, name=None):
    """Computes binary cross entropy given `preds`.

    For brevity, let `x = `, `z = targets`.  The logistic loss is

        loss(x, z) = - sum_i (x[i] * log(z[i]) + (1 - x[i]) * log(1 - z[i]))

    Args:
        preds: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `preds`.
    """
    eps = 1e-12
    with ops.op_scope([preds, targets], name, "bce_loss") as name:
        preds = ops.convert_to_tensor(preds, name="preds")
        targets = ops.convert_to_tensor(targets, name="targets")
        return tf.reduce_mean(-(targets * tf.log(preds + eps) +
                              (1. - targets) * tf.log(1. - preds + eps)))

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d", no_summery=False):

    if(no_summery):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            res = tf.nn.relu(conv)
    else:
        with tf.variable_scope(name):
            with tf.name_scope('weights'):
                w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                variable_summaries(w)
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

            with tf.name_scope('biases'):
                biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
                variable_summaries(biases)

            with tf.name_scope('Wx_plus_b'):
                conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
                tf.summary.histogram('pre_activations',conv)

            res = tf.nn.relu(conv)
            tf.summary.histogram('activations',res)

        # res = conv
    return res

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False, no_summery=False):
    if(no_summery):
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))

            # try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])
            # Support for verisons of TensorFlow before 0.7.0
            # except AttributeError:
            #     deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
            #                         strides=[1, d_h, d_w, 1])

            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
            res = tf.nn.relu(deconv)
            # res = deconv
            if with_w:
                return res, w, biases
            else:
                return res
    else:
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            with tf.name_scope('weights'):
                w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                    initializer=tf.random_normal_initializer(stddev=stddev))
                variable_summaries(w)

            # try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])
            # Support for verisons of TensorFlow before 0.7.0
            # except AttributeError:
            #     deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
            #                         strides=[1, d_h, d_w, 1])

            with tf.name_scope('biases'):
                biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
                variable_summaries(biases)

            with tf.name_scope('Wx_plus_b'):
                deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
                tf.summary.histogram('pre_activations',deconv)

            res = tf.nn.relu(deconv)
            tf.summary.histogram('activations',res)
            # res = deconv
            if with_w:
                return res, w, biases
            else:
                return res
       

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False, no_summery=False):
    shape = input_.get_shape().as_list()

    if(no_summery):
        with tf.variable_scope(scope or "Linear"):
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
            bias = tf.get_variable("bias", [output_size],
                    initializer=tf.constant_initializer(bias_start))
            ll = tf.matmul(input_, matrix) + bias
            res = lrelu(ll)
            # res = tf.matmul(input_,matrix) + bias
            if with_w:
                return res, matrix, bias
            else:
                return res
    else:
        with tf.variable_scope(scope or "Linear"):
            with tf.name_scope('weights'):
                matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                         tf.random_normal_initializer(stddev=stddev))
                variable_summaries(matrix)

            with tf.name_scope('biases'):
                bias = tf.get_variable("bias", [output_size],
                    initializer=tf.constant_initializer(bias_start))
                variable_summaries(bias)

            with tf.name_scope('Wx_plus_b'):
                ll = tf.matmul(input_, matrix) + bias
                tf.summary.histogram('pre_activations',ll)

            res = lrelu(ll)
            tf.summary.histogram('activations',res)
            # res = tf.matmul(input_,matrix) + bias
            if with_w:
                return res, matrix, bias
            else:
                return res


def max_pool_3x3_2(x,name="pool_3x3_2"):
    """max_pool_3x3 downsamples a feature map by 2X."""
    with tf.variable_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3_2t(x,name="pool_3x3_2t"):
    """max_pool_3x3 downsamples a feature map by 2X."""
    with tf.variable_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                              strides=[1, 2, 1, 1], padding='SAME')

def conv2d_valid(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d_valid",no_summery=False):
    if(no_summery):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))

            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            res = tf.nn.relu(conv)
            # res = conv
            return res
    else:
        with tf.variable_scope(name):
            with tf.name_scope('weights'):
                w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                    initializer=tf.truncated_normal_initializer(stddev=stddev))
                variable_summaries(w)


            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')

            with tf.name_scope('biases'):
                biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
                variable_summaries(biases)

            with tf.name_scope('Wx_plus_b'):
                conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
                tf.summary.histogram('pre_activations',conv)

            res = tf.nn.relu(conv)
            tf.summary.histogram('activations',res)
            # res = conv

            return res

def deconv2d_valid(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d_valid", with_w=False,no_summery=False):
    if(no_summery):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))

                # trym
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1], padding="VALID")

            # Support for verisons of TensorFlow before 0.7.0
            # except AttributeError:
            #     deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
            #                             strides=[1, d_h, d_w, 1])

            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
            res = tf.nn.relu(deconv)
            #
            # res = deconv
            if with_w:
                return res, w, biases
            else:
                return res
    else:
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            with tf.name_scope('weights'):
                w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                    initializer=tf.random_normal_initializer(stddev=stddev))
                variable_summaries(w)

            # try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                                strides=[1, d_h, d_w, 1],padding="VALID")

            # Support for verisons of TensorFlow before 0.7.0
            # except AttributeError:
            #     deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
            #                             strides=[1, d_h, d_w, 1])

            with tf.name_scope('biases'):
                biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
                variable_summaries(biases)

            with tf.name_scope('Wx_plus_b'):
                deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
                tf.summary.histogram('pre_activations',deconv)

            res = tf.nn.relu(deconv)
            tf.summary.histogram('activations',res)
            #
            # res = deconv
            if with_w:
                return res, w, biases
            else:
                return res


def variable_summaries(var):
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)
