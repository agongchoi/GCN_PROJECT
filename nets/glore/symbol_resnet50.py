# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import mxnet as mx
from symbol_block import *
from symbol_glore import GloRe_Unit

k_sec  = {  2:  3, \
            3:  4, \
            4:  6, \
            5:  3  }

def get_before_pool():
    data = mx.symbol.Variable(name="data")

    # conv1
    conv1_x = Conv(data=data,  num_filter=64,  kernel=(7,7), name='conv1', pad=(3,3), stride=(2,2))
    conv1_x = BN_AC(data=conv1_x, name='conv1')
    conv1_x = mx.symbol.Pooling(data=conv1_x, pool_type="max", kernel=(3, 3), pad=(1,1), stride=(2,2), name="pool1")

    # conv2
    num_in  = 32
    num_mid = 64
    num_out = 256
    for i in range(1,k_sec[2]+1):
        conv2_x = Residual_Unit(data=(conv1_x if i==1 else conv2_x),
                                num_in=(num_in if i==1 else num_out),
                                num_mid=num_mid,
                                num_out=num_out,
                                name="conv2_B%02d"%i,
                                first_block=(i==1), stride=((1,1) if (i==1) else (1,1)))

    # conv3
    num_in  = num_out
    num_mid = int(2*num_mid)
    num_out = int(2*num_out)
    for i in range(1,k_sec[3]+1):
        conv3_x = Residual_Unit(data=(conv2_x if i==1 else conv3_x),
                                num_in=(num_in if i==1 else num_out),
                                num_mid=num_mid,
                                num_out=num_out,
                                name="conv3_B%02d"%i,
                                first_block=(i==1), stride=((2,2) if (i==1) else (1,1)))
        if i in [1,3]:
            conv3_x = GloRe_Unit(data=conv3_x,
                                 settings=(num_out, num_mid),
                                 name="conv3_B%02d_extra"%i, stride=(1,1))

    # conv4
    num_in  = num_out
    num_mid = int(2*num_mid)
    num_out = int(2*num_out)
    for i in range(1,k_sec[4]+1):
        conv4_x = Residual_Unit(data=(conv3_x if i==1 else conv4_x),
                                num_in=(num_in if i==1 else num_out),
                                num_mid=num_mid,
                                num_out=num_out,
                                name="conv4_B%02d"%i,
                                first_block=(i==1), stride=((2,2) if (i==1) else (1,1)))
        if i in [1,3,5]:
            conv4_x = GloRe_Unit(data=conv4_x,
                                 settings=(num_out, num_mid),
                                 name="conv4_B%02d_extra"%i, stride=(1,1))

    # conv5
    num_in  = num_out
    num_mid = int(2*num_mid)
    num_out = int(2*num_out)
    for i in range(1,k_sec[5]+1):
        conv5_x = Residual_Unit(data=(conv4_x if i==1 else conv5_x),
                                num_in=(num_in if i==1 else num_out),
                                num_mid=num_mid,
                                num_out=num_out,
                                name="conv5_B%02d"%i,
                                first_block=(i==1), stride=((2,2) if (i==1) else (1,1)))

    # output
    conv5_x = BN_AC(conv5_x, name='tail')
    return conv5_x


def get_linear(num_classes = 1000):
    before_pool = get_before_pool()
    # - - - - -
    pool5     = mx.symbol.Pooling(data=before_pool, pool_type="avg", kernel=(7, 7), stride=(1,1), name="global-pool")
    flat5     = mx.symbol.Flatten(data=pool5, name='flatten')
    fc6       = mx.symbol.FullyConnected(data=flat5, num_hidden=num_classes, name='classifier')
    return fc6


def get_symbol(num_classes = 1000):
    fc6       = get_linear(num_classes)
    softmax   = mx.symbol.SoftmaxOutput( data=fc6,  name='softmax')
    sys_out   = softmax
    return sys_out
