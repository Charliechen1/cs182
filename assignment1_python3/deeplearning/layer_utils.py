from deeplearning.layers import *
from deeplearning.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_bn_relu_dropout_forward(x, w, b, gamma, beta, bn_param, dropout_param):
    # {affine - [batch norm] - relu - [dropout]}
    z, aff_cache = affine_forward(x, w, b)
    z, bn_cache = batchnorm_forward(z, gamma, beta, bn_param)
    z, relu_cache = relu_forward(z)
    out, drop_cache = dropout_forward(z, dropout_param)  
    cache = aff_cache, bn_cache, relu_cache, drop_cache
    return out, cache
    
def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def affine_bn_relu_dropout_backward(dout, cache):
    aff_cache, bn_cache, relu_cache, drop_cache = cache
    dz = dropout_backward(dout, drop_cache)
    dz = relu_backward(dz, relu_cache)
    dz, dgamma, dbeta = batchnorm_backward_alt(dz, bn_cache)
    dx, dw, db = affine_backward(dz, aff_cache)
    return dx, dw, db, dgamma, dbeta

def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_bn_forward(x, w, b, gamma, beta, conv_param, pool_param, bn_param):
    """
    Convenience layer that performs a convolution, a ReLU, a pool and a spatial batch normalization.
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    s, pool_cache = max_pool_forward_fast(s, pool_param)
    out, bn_cache = spatial_batchnorm_forward(s, gamma, beta, bn_param)
    cache = (conv_cache, relu_cache, pool_cache, bn_cache)
    return out, cache

def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db

def conv_relu_pool_bn_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache, bn_cache = cache
    da, dgamma, dbeta = spatial_batchnorm_backward(dout, bn_cache)
    da = max_pool_backward_fast(da, pool_cache)
    da = relu_backward(da, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta