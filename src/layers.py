# File: layers.py
# Author: Qian Ge <geqian1001@gmail.com>
# Reference code: https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/models/

import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope
import numpy as np

@add_arg_scope
def conv(x, filter_size, out_dim, 
         name='conv', namew='conv', nameb = 'conv', stride=2, 
         padding='SAME',
         nl=tf.nn.relu,
         data_dict=None,
         init_w=None, init_b=None,
         use_bias=True,
         wd=None,
         trainable=True,
         mode="train",
         graph=None):
    """ 
    2D convolution 

    Args:
        x (tf.tensor): a 4D tensor
           Input number of channels has to be known
        filter_size (int or list with length 2): size of filter
        out_dim (int): number of output channels
        name (str): name scope of the layer
        stride (int or list): stride of filter
        padding (str): 'VALID' or 'SAME' 
        init_w, init_b: initializer for weight and bias variables. 
           Default to 'random_normal_initializer'
        nl: a function

    Returns:
        tf.tensor with name 'output'
    """

    in_dim = int(x.shape[-1])
    assert in_dim is not None,\
    'Number of input channel cannot be None!'

    filter_shape = get_shape2D(filter_size) + [in_dim, out_dim]
    strid_shape = get_shape4D(stride)

    padding = padding.upper()

    convolve = lambda i, k: tf.nn.conv2d(i, k, strid_shape, padding)

    weights = new_weights(namew, filter_shape, initializer=init_w,
                            mode=mode, graph=graph, trainable=trainable, wd=wd)
    out = convolve(x, weights)

    if use_bias:
        biases = new_biases(nameb, [out_dim], initializer=init_b,
                        mode=mode, graph=graph, trainable=trainable)
        out = tf.nn.bias_add(out, biases)
    
    output = nl(out)
    return output

@add_arg_scope
def conv3d(x, filter_size, out_dim, 
            name='conv', namew = 'conv', nameb='conv', stride=1, 
            padding='SAME',
            nl=tf.nn.relu,
            data_dict=None,
            init_w=None, init_b=None,
            use_bias=True,
            wd=None,
            trainable=True,
            mode="train",
            graph=None):
    """ 
    3D convolution 

    Args:
        x (tf.tensor): a 5D tensor
           Input number of channels has to be known
        filter_size (int or list with length 2): size of filter
        out_dim (int): number of output channels
        name (str): name scope of the layer
        stride (int or list): stride of filter
        padding (str): 'VALID' or 'SAME' 
        init_w, init_b: initializer for weight and bias variables. 
           Default to 'random_normal_initializer'
        nl: activation function

    Returns:
        tf.tensor with name 'output'
    """

    in_dim = int(x.shape[4])
    assert in_dim is not None,\
    'Number of input channel cannot be None!'

    filter_shape = get_shape3D(filter_size) + [in_dim, out_dim]
    strid_shape = get_shape5D(stride)

    padding = padding.upper()

    convolve = lambda i, k: tf.nn.conv3d(i, k, strid_shape, padding)

    weights = new_weights(namew, filter_shape, initializer=init_w,
                            mode=mode, graph=graph, trainable=trainable, wd=wd)
    out = convolve(x, weights)

    if use_bias:
        biases = new_biases(nameb, [out_dim], initializer=init_b,
                        mode=mode, graph=graph, trainable=trainable)
        out = tf.nn.bias_add(out, biases)
    
    output = nl(out)
    return output

@add_arg_scope
def fc(x, out_dim, name='fc', namew='fc', nameb='fc', nl=tf.identity, 
       init_w=None, init_b=None,
       data_dict=None,
       wd=None, 
       trainable=True,
       re_dict=False,
       mode="train",
       graph=None):
    """ 
    Fully connected layer 

    Args:
        x (tf.tensor): a tensor to be flattened 
           The first dimension is the batch dimension
        num_out (int): dimension of output
        name (str): name scope of the layer
        init: initializer for variables. Default to 'random_normal_initializer'
        nl: a function

    Returns:
        tf.tensor with name 'output'
    """

    # x_flatten = batch_flatten(x)
    # x_shape = x_flatten.get_shape().as_list()
    in_dim = x.get_shape().as_list()[1]

    with tf.variable_scope(name) as scope:
        weights = new_weights(namew, [in_dim, out_dim], initializer=init_w,
                              mode=mode, graph=graph, trainable=trainable, wd=wd)
        biases = new_biases(nameb, [out_dim], initializer=init_b,
                            mode=mode, graph=graph, trainable=trainable)
        act = tf.nn.xw_plus_b(x, weights, biases)

        output = nl(act)
        if re_dict is True:
            return {'outputs': output, 'weights': weights, 'biases': biases}
        else:
            return output

def max_pool(x, name='max_pool', filter_size=2, stride=None, padding='VALID'):
    """ 
    Max pooling layer 

    Args:
        x (tf.tensor): a tensor 
        name (str): name scope of the layer
        filter_size (int or list with length 2): size of filter
        stride (int or list with length 2): Default to be the same as shape
        padding (str): 'VALID' or 'SAME'. Use 'SAME' for FCN.

    Returns:
        tf.tensor with name 'name'
    """

    padding = padding.upper()
    filter_shape = get_shape4D(filter_size)
    if stride is None:
        stride = filter_shape
    else:
        stride = get_shape4D(stride)

    return tf.nn.max_pool(x, ksize=filter_shape, 
                          strides=stride, 
                          padding=padding, name=name)

def max_pool3d(x, name='max_pool', filter_size=2, stride=None, padding='SAME'):
    """ 
    Max pooling layer 

    Args:
        x (tf.tensor): a tensor 
        name (str): name scope of the layer
        filter_size (int or list with length 3): size of filter
        stride (int or list with length 3): Default to be the same as shape
        padding (str): 'VALID' or 'SAME'. Use 'SAME' for FCN.

    Returns:
        tf.tensor with name 'name'
    """

    padding = padding.upper()
    filter_shape = get_shape5D(filter_size)
    if stride is None:
        stride = filter_shape
    else:
        stride = get_shape5D(stride)

    return tf.nn.max_pool3d(x, ksize=filter_shape, 
                            strides=stride, 
                            padding=padding, name=name)

def global_avg_pool(x, name='global_avg_pool', data_format='NHWC'):
    assert x.shape.ndims == 4
    assert data_format in ['NHWC', 'NCHW']
    with tf.name_scope(name):
        axis = [1, 2] if data_format == 'NHWC' else [2, 3]
        return tf.reduce_mean(x, axis)

def dropout(x, keep_prob, is_training, name='dropout'):
    """ 
    Dropout 

    Args:
        x (tf.tensor): a tensor 
        keep_prob (float): keep prbability of dropout
        is_training (bool): whether training or not
        name (str): name scope

    Returns:
        tf.tensor with name 'name'
    """

    # tf.nn.dropout does not have 'is_training' argument
    return tf.nn.dropout(x, keep_prob)
    #return tf.layers.dropout(x, rate=1 - keep_prob, training=is_training, name=name)
    

def batch_norm(x, train=True, name='bn'):
    """ 
    batch normal 

    Args:
        x (tf.tensor): a tensor 
        name (str): name scope
        train (bool): whether training or not

    Returns:
        tf.tensor with name 'name'
    """
    return tf.contrib.layers.batch_norm(x, decay=0.9, 
                          updates_collections=None,
                          epsilon=1e-5, scale=False,
                          is_training=train, scope=name)

def leaky_relu(x, leak=0.2, name='LeakyRelu'):
    """ 
    leaky_relu 
        Allow a small non-zero gradient when the unit is not active

    Args:
        x (tf.tensor): a tensor 
        leak (float): Default to 0.2

    Returns:
        tf.tensor with name 'name'
    """
    return tf.maximum(x, leak*x, name=name)

def new_normal_variable(name, shape=None, trainable=True, stddev=0.002):
    return tf.get_variable(name, shape=shape, trainable=trainable, 
                 initializer=tf.random_normal_initializer(stddev=stddev))

def new_variable(name, idx, shape, initializer=None):
    var = tf.get_variable(name, shape=shape, 
                           initializer=initializer) 

    return var

def new_weights(name, shape, initializer=None, wd=None, 
                mode="train",graph=None,
                trainable=True): 
    if graph!=None:
        # assert graph != None, "\033[31m in test mode, you must provide the graph\033[0m"
        var = graph.get_tensor_by_name(name+":0")
        print('{} restore from pretrained model...'.format(name))
    elif wd is not None:
        print('Random init {} weights with weight decay...'.format(name))
        if initializer is None:
            initializer = tf.truncated_normal_initializer(stddev=0.01)
        var = tf.get_variable(name, shape=shape, 
                                  initializer=initializer,
                                  trainable=trainable) 
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    else:
        print('Random init {} weights...'.format(name))
        if initializer is None:
            initializer = tf.random_normal_initializer(stddev=0.002)
        var = tf.get_variable(name, shape=shape, 
                            initializer=initializer,
                            trainable=trainable) 
    return var

def new_biases(name, shape, initializer=None,
                mode="train",graph=None,
                trainable=True):
    if graph!=None:
        # assert graph != None, "\033[31m in test mode, you must provide the graph\033[0m"
        var = graph.get_tensor_by_name(name)
        print('{} restore from pretrained model...'.format(name))
    else:
        print('Random init {} biases...'.format(name))
        if initializer is None:
            initializer = tf.random_normal_initializer(stddev=0.002)
        var = tf.get_variable(name, shape=shape, 
                           initializer=initializer,
                           trainable=trainable) 
    return var



def get_shape2D(in_val):
    """
    Return a 2D shape 

    Args:
        in_val (int or list with length 2) 

    Returns:
        list with length 2
    """
    if isinstance(in_val, int):
        return [in_val, in_val]
    if isinstance(in_val, list):
        assert len(in_val) == 2
        return in_val
    raise RuntimeError('Illegal shape: {}'.format(in_val))

def get_shape3D(in_val):
    """
    Return a 3D shape 

    Args:
        in_val (int or list with length 3) 

    Returns:
        list with length 3
    """
    if isinstance(in_val, int):
        return [in_val, in_val, in_val]
    if isinstance(in_val, list):
        assert len(in_val) == 3, "\033[31m 3d filter shape must have rank 3"
        return in_val
    raise RuntimeError('Illegal shape: {}'.format(in_val))

def get_shape4D(in_val):
    """
    Return a 4D shape

    Args:
        in_val (int or list with length 2)

    Returns:
        list with length 4
    """
    # if isinstance(in_val, int):
    return [1] + get_shape2D(in_val) + [1]

def get_shape5D(in_val):
    """
    Return a 4D shape

    Args:
        in_val (int or list with length 2)

    Returns:
        list with length 4
    """
    # if isinstance(in_val, int):
    return [1] + get_shape3D(in_val) + [1]

def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))


