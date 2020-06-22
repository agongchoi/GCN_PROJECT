# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import mxnet as mx
from symbol_basic import *


def Residual_Unit(data, num_in, num_mid, num_out, name, first_block=False, stride=(1, 1), g=1):

    # main part
    conv_m1 = BN_AC_Conv(data=data,    num_filter=num_mid, kernel=(1, 1), pad=(0, 0), name=('%s_conv-m1' % name))
    conv_m2 = BN_AC_Conv(data=conv_m1, num_filter=num_mid, kernel=(3, 3), pad=(1, 1), name=('%s_conv-m2' % name), stride=stride, num_group=g)
    conv_m3 = BN_AC_Conv(data=conv_m2, num_filter=num_out, kernel=(1, 1), pad=(0, 0), name=('%s_conv-m3' % name))

    # adapter
    if first_block:
        data  = BN_AC_Conv(data=data,  num_filter=num_out, kernel=(1, 1), pad=(0, 0), name=('%s_conv-w1' % name), stride=stride)

    out   = mx.symbol.ElementWiseSum(*[data, conv_m3],                                name=('%s_sum' % name))

    return out
