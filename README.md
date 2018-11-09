# unpool
Implement unpool operation(https://arxiv.org/abs/1506.02351) in tensorflow.

Use tf.nn.max_pool_with_argmax(https://tensorflow.google.cn/api_docs/python/tf/nn/max_pool_with_argmax) to pool to get argmax.

NOTE: 1.This code is UNTESTED, may have some BUGs!
      2.Because of using tf.scatter_update, this operation will cause tensorflow can not be able to automatically compute the gradient.(see https://github.com/tensorflow/tensorflow/issues/2770)
