# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import mxnet as mx
import logging

bn_momentum = 0.92

def BK(data):
    return mx.symbol.BlockGrad(data=data)

# - - - - - - - - - - - - - - - - - - - - - - -
# Fundamental Elements
def BN(data, fix_gamma=False, momentum=bn_momentum, name=None):
    bn     = mx.symbol.BatchNorm( data=data, fix_gamma=fix_gamma, momentum=bn_momentum, eps=0.0001, name=('%s_bn'%name))
    return bn

def AC(data, act_type='relu', name=None):
    act    = mx.symbol.Activation(data=data, act_type=act_type, name=('%s_%s' % (name, act_type)))
    return act

def BN_AC(data, momentum=bn_momentum, name=None):
    bn     = BN(data=data, name=name, fix_gamma=False, momentum=momentum)
    bn_ac  = AC(data=bn,   name=name)
    return bn_ac

def Conv(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, no_bias=True, attr=None, num_group=1):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=('%s_conv' %name), no_bias=no_bias, attr=attr)
    return conv

# - - - - - - - - - - - - - - - - - - - - - - -
# Standard Common functions < CVPR >
def Conv_BN(   data, num_filter,  kernel, pad, stride=(1,1), name=None, no_bias=True, attr=None, num_group=1):
    cov    = Conv(   data=data,   num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, no_bias=no_bias, attr=attr)
    cov_bn = BN(     data=cov,    name=name)
    return cov_bn

def Conv_BN_AC(data, num_filter,  kernel, pad, stride=(1,1), name=None, no_bias=True, attr=None, num_group=1):
    cov_bn = Conv_BN(data=data,   num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, no_bias=no_bias, attr=attr)
    cov_ba = AC(     data=cov_bn, name=name)
    return cov_ba

# - - - - - - - - - - - - - - - - - - - - - - -
# Standard Common functions < ECCV >
def BN_Conv(   data, num_filter,  kernel, pad, stride=(1,1), name=None, no_bias=True, attr=None, num_group=1):
    bn     = BN(     data=data,   name=name, fix_gamma=True)
    bn_cov = Conv(   data=bn,     num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, no_bias=no_bias, attr=attr)
    return bn_cov

def AC_Conv(   data, num_filter,  kernel, pad, stride=(1,1), name=None, no_bias=True, attr=None, num_group=1):
    ac     = AC(     data=data,   name=name)
    ac_cov = Conv(   data=ac,     num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, no_bias=no_bias, attr=attr)
    return ac_cov

def BN_AC_Conv(data, num_filter,  kernel, pad, stride=(1,1), name=None, no_bias=True, attr=None, num_group=1):
    bn     = BN(     data=data,   name=name)
    ba_cov = AC_Conv(data=bn,     num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name, no_bias=no_bias, attr=attr)
    return ba_cov
