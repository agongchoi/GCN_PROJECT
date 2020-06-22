# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import mxnet as mx
from symbol_basic import *

# - - - - - - - - - - - - - - - - - - - - - - -
# GloRe Unit
def GloRe_Unit(data, settings, name, stride=(1,1)):
    num_in, num_mid = settings
    num_state = int(2 * num_mid)
    num_node  = int(1 * num_mid)

    # reduce sampling space if is required
    if tuple(stride) == (1,1):
        x_pooled = data  # default
    else:
        x_pooled = mx.symbol.Pooling(data=data, pool_type="avg", kernel=stride, stride=stride, name=('%s_pooling' % name))

    # generate "state" for each node:
    # (N, num_in, H_sampled, W_sampled) --> (N, num_state, H_sampled, W_sampled)
    #                                   --> (N, num_state, H_sampled*W_sampled)
    x_state = BN_AC_Conv(data=x_pooled, num_filter=num_state,  kernel=(1, 1), pad=(0, 0), name=('%s_conv-state' % name))
    x_state_reshaped = mx.symbol.reshape(x_state, shape=(0, 0, -1),                       name=('%s_conv-state-reshape' % name))

    # prepare "projection" function (coordinate space -> interaction space):
    # (N, num_in, H_sampled, W_sampled) --> (N, num_node, H_sampled, W_sampled)
    #                                   --> (N, num_node, H_sampled*W_sampled)
    x_proj = BN_AC_Conv(data=x_pooled, num_filter=num_node, kernel=(1, 1), pad=(0, 0),   name=('%s_conv-proj' % name))
    x_proj_reshaped = mx.symbol.reshape(x_proj, shape=(0, 0, -1),                        name=('%s_conv-proj-reshape' % name))

    # prepare "reverse projection" function (interaction space -> coordinate space)
    # (N, num_in, H, W) --> (N, num_node, H, W)
    #                   --> (N, num_node, H*W)
    x_rproj_reshaped = x_proj_reshaped
 
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Projection: coordinate space -> interaction space
    # (N, num_state, H_sampled*W_sampled) x (N, num_node, H_sampled*W_sampled)T --> (N, num_state, num_node)
    x_n_state = mx.symbol.batch_dot(lhs=x_state_reshaped, rhs=x_proj_reshaped, transpose_b=True,  name=('%s_proj' % name))

    # ------
    # Relation Reasoning    
    # -> propogate information: [G * X]
    #     (N, num_state, num_node)
    #     -permute-> (N, num_node, num_state) 
    #     --conv-->  (N, num_node, num_state) 
    #     -permute-> (N, num_state, num_node) (+ shortcut)
    #     --conv-->  (N, num_state, num_node)
    x_n_rel = x_n_state
    x_n_rel = mx.symbol.transpose(data=x_n_rel, axes=(0,2,1),                           name=('%s_GCN-G-permute1' % name))
    x_n_rel = BN_AC_Conv(data=x_n_rel, num_filter=num_node,  kernel=1, pad=0, stride=1, name=('%s_GCN-G' % name))
    x_n_rel = mx.symbol.transpose(data=x_n_rel, axes=(0,2,1),                           name=('%s_GCN-G-permute2' % name))
    # -> add shortcut
    x_n_rel = mx.symbol.ElementWiseSum(*[x_n_state, x_n_rel],                           name=('%s_GCN-G_sum' % name))
    # -> update state: [H * W]
    x_n_rel = BN_AC_Conv(data=x_n_rel, num_filter=num_state, kernel=1, pad=0, stride=1, name=('%s_GCN-GHW' % name))
    # -> output    
    x_n_state_new = x_n_rel

    # ------
    # Reverse Projection: interaction space -> coordinate space
    # (N, num_state, num_node) x (N, num_node, H*W) --> (N, num_state, H*W)
    #                                               --> (N, num_state, H, W)
    x_out = mx.symbol.batch_dot(lhs=x_n_state_new, rhs=x_rproj_reshaped, name=('%s_reverse-proj' % name))
    x_out = mx.symbol.reshape_like(x_out, rhs=x_state,                   name=('%s_reverse-proj-reshape' % name))

    # ------
    # extend dimension
    x_out = BN_AC_Conv(data=x_out, num_filter=num_in, kernel=(1, 1), pad=(0, 0), name=('%s_fc-2' % name))

    out   = mx.symbol.ElementWiseSum(*[data, x_out], name=('%s_sum' % name))

    return out
