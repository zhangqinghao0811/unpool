# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:44:45 2018

@author: 张庆昊
"""

import numpy as np

def unpool(x, argmax, strides, unpool_shape=None, batch_size=None, name='unpool'):
    x_shape = x.get_shape().as_list()
    argmax_shape = argmax.get_shape().as_list()
    assert not(x_shape[0] is None and batch_size is None), "must input batch_size if number of batch is alterable"
    if x_shape[0] is None:
        x_shape[0] = batch_size
    if argmax_shape[0] is None:
        argmax_shape[0] = x_shape[0]
    if unpool_shape is None:
        unpool_shape = [x_shape[i] * strides[i] for i in range(4)]
    unpool = tf.get_variable(name=name, shape=[np.prod(unpool_shape)], initializer=tf.zeros_initializer(), trainable=False)
    argmax = tf.cast(argmax, tf.int32)
    argmax = tf.reshape(argmax, [np.prod(argmax_shape)])
    x = tf.reshape(x, [np.prod(x_shape)])
    x = tf.reshape(x, [np.prod(argmax.get_shape().as_list())])
    unpool = tf.scatter_update(unpool, argmax, x)
    unpool = tf.reshape(unpool, unpool_shape)
    return unpool
