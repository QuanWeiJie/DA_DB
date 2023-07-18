# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr
from db_fpn_domain import Global_DomainAdaptionModule,DAImgHead2

class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=False)

        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=act,
            param_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1e-4)))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DeConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(DeConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.deconv = nn.Conv2DTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=False)
        self.bn = nn.BatchNorm(
            num_channels=out_channels,
            act=act,
            param_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1e-4))
            )

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        return x
class EAST_Global_DomainAdaptionModule(Global_DomainAdaptionModule):
    def __init__(self):
        super().__init__()
        self.imgHead =  DAImgHead2(512)

class EASTFPN_domain(nn.Layer):
    def __init__(self, in_channels, model_name, **kwargs):
        super().__init__()
        self.model_name = model_name
        if self.model_name == "large":
            self.out_channels = 128
        else:
            self.out_channels = 64
        self.in_channels = in_channels[::-1]
        self.h1_conv = ConvBNLayer(
            in_channels=self.out_channels+self.in_channels[1],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu')
        self.h2_conv = ConvBNLayer(
            in_channels=self.out_channels+self.in_channels[2],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu')
        self.h3_conv = ConvBNLayer(
            in_channels=self.out_channels+self.in_channels[3],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu')
        self.g0_deconv = DeConvBNLayer(
            in_channels=self.in_channels[0],
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            if_act=True,
            act='relu')
        self.g1_deconv = DeConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            if_act=True,
            act='relu')
        self.g2_deconv = DeConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            if_act=True,
            act='relu')
        self.g3_conv = ConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu')
        self.Global_DAmodule = EAST_Global_DomainAdaptionModule() 

    def forward(self, x):
        f = x[::-1]  # ::-1 取反

        h = f[0]  # 在这里加 8*512*16*16
        global_domain_cls = self.Global_DAmodule(h)
        g = self.g0_deconv(h)
        h = paddle.concat([g, f[1]], axis=1)
        h = self.h1_conv(h)
        g = self.g1_deconv(h)
        h = paddle.concat([g, f[2]], axis=1)
        h = self.h2_conv(h)
        g = self.g2_deconv(h)
        h = paddle.concat([g, f[3]], axis=1)
        h = self.h3_conv(h)
        g = self.g3_conv(h)
        
        return {'g': g,'global_domain_cls':global_domain_cls}