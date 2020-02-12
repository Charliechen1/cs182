import numpy as np

from deeplearning.layers import *
from deeplearning.layer_utils import *

class MiniResNet(object):
    def __init__(self, input_dim=(3, 32, 32), 
                 conv_params=[{'num_filters': 32, 'filter_size': 7, 'stride': 1}],
                 hidden_dims=[100],
                 pool_params=[{'pool_height': 2, 'pool_width': 2, 'stride': 2}],
                 num_conv=1,
                 num_affi=1,
                 num_classes=10, 
                 weight_scale=1e-3, 
                 reg=0.0,
                 dtype=np.float32, 
                 dropout=0.0, 
                 use_batchnorm=False, 
                 seed=None):

        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = num_conv + num_affi + 1
        self.num_conv = num_conv
        self.num_affi = num_affi
        self.dtype = dtype
        self.conv_params = conv_params
        self.pool_params = pool_params
        self.hidden_dims = hidden_dims
        self.params = {}

        # some checking
        assert(len(conv_params) == num_conv)
        assert(len(hidden_dims) == num_affi)
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        C, H, W = input_dim
        idx, conv_idx, aff_idx, pool_idx = 0, 0, 0, 0
        for _ in range(num_conv):
            idx += 1
            conv_param = conv_params[conv_idx]
            conv_idx += 1
            pool_param = pool_params[pool_idx]
            pool_idx += 1
            
            num_filters, filter_size, conv_stride, pad = \
                conv_param['num_filters'], conv_param['filter_size'], conv_param['stride'], conv_param['pad']
            pool_height, pool_width, pool_stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
            
            self.params[f'W{idx}'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
            self.params[f'b{idx}'] = np.zeros(num_filters)
            
            # update the dimentions for next conv layer
            C = num_filters
            H = 1 + (H + 2 * pad - filter_size) // conv_stride
            W = 1 + (W + 2 * pad - filter_size) // conv_stride

            H = 1 + (H - pool_height) // pool_stride
            W = 1 + (W - pool_width) // pool_stride
        
        affi_dims = [H * W * num_filters] + hidden_dims + [num_classes]
        
        for i in range(len(affi_dims) - 1):
            idx += 1
            self.params[f"W{idx}"] = np.random.normal(0, weight_scale, (affi_dims[i], affi_dims[i + 1]))
            self.params[f"b{idx}"] = np.zeros(affi_dims[i + 1])
        
        idx = 0
        if self.use_batchnorm:
            #for _ in conv_params:
            #    idx += 1
            #    self.params[f"beta{idx}"] = np.zeros(C)
            #    self.params[f"gamma{idx}"] = np.ones(C)
            
            for dim in hidden_dims:
                idx += 1
                self.params[f"beta{idx}"] = np.zeros(dim)
                self.params[f"gamma{idx}"] = np.ones(dim)
        #print("parameters:")
        #for k, v in self.params.items():
        #    print(k, v.shape)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.dropout_param is not None:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        idx = 0
        conv_caches, affi_caches, drop_caches, bn_caches = [], [], [], []
        z = X
        for c_i in range(self.num_conv):
            idx += 1
            z, conv_cache = conv_relu_pool_forward(z, 
                       self.params[f"W{idx}"], 
                       self.params[f"b{idx}"], 
                       self.conv_params[c_i], 
                       self.pool_params[c_i])
            conv_caches.append(conv_cache)
            #if self.use_batchnorm:
            #    z, spbn_cache = spatial_batchnorm_forward(z, self.params[f"gamma{idx}"], self.params[f"beta{idx}"], bn_param[idx - 1])
            #    bn_caches.append(spbn_cache)
        # record the shape of z here, need to recover the shape when doing back propagation
        interim_shape = z.shape
        z = z.reshape(z.shape[0], -1)
        for a_i in range(self.num_affi):
            idx += 1
            z, affi_cache = affine_bn_relu_dropout_forward(z, 
                       self.params[f"W{idx}"], 
                       self.params[f"b{idx}"], 
                       self.params[f"gamma{a_i + 1}"], 
                       self.params[f"beta{a_i + 1}"], 
                       #self.bn_params[idx - 1], 
                       self.bn_params[a_i], 
                       self.dropout_param)
            affi_caches.append(affi_cache)
        
        scores, last_cache = affine_forward(z, self.params[f"W{self.num_layers}"], self.params[f"b{self.num_layers}"])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores
        
        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        for idx in range(1, self.num_layers + 1):
            loss += 0.5 * self.reg * (np.linalg.norm(self.params[f"W{idx}"]) ** 2)
        
        dz, grads[f"W{self.num_layers}"], grads[f"b{self.num_layers}"] = affine_backward(dout, last_cache)
        idx = self.num_layers
        for a_i in range(self.num_affi - 1, -1, -1):
            idx -= 1
            # dx, dw, db, dgamma, dbeta
            dz, grads[f'W{idx}'], grads[f'b{idx}'], grads[f'gamma{a_i + 1}'], grads[f'beta{a_i + 1}'] \
                = affine_bn_relu_dropout_backward(dz, affi_caches[a_i])
        dz = dz.reshape(interim_shape)
        for c_i in range(self.num_conv - 1, -1, -1):
            idx -= 1
            dz, grads[f'W{idx}'], grads[f'b{idx}'] = conv_relu_pool_backward(dz, conv_caches[c_i])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        return loss, grads
