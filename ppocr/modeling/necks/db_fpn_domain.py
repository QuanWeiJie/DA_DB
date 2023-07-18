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
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))


import math
from paddle import ParamAttr
from paddle.autograd import PyLayer

# def get_bias_attr(k):
#     stdv = 1.0 / math.sqrt(k * 1.0)
#     initializer = paddle.nn.initializer.Uniform(-stdv, stdv)
#     bias_attr = ParamAttr(initializer=initializer)
#     return bias_attr

class _GradientScalarLayer(PyLayer):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight*grad_input

gradient_scalar = _GradientScalarLayer.apply


class DAImgHead(nn.Layer):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self,in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.conv1_da = nn.Conv2D(in_channels, in_channels//4, kernel_size=1, stride=1,
                        weight_attr=ParamAttr(initializer=weight_attr),
                        bias_attr=False)
        self.conv_bn1 = nn.BatchNorm(
                        num_channels=in_channels // 4,
                        param_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)),
                        bias_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1e-4)),
                        act='relu')
        self.conv2_da = nn.Conv2D(in_channels//4, 1, kernel_size=1, stride=1,
                        weight_attr=ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()),
                        bias_attr=False)
        
    def forward(self, x):
        
        t = self.conv_bn1(self.conv1_da(x))
        return self.conv2_da(t)

class DAImgHead2(nn.Layer):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self,in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead2, self).__init__()
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.conv1_da = nn.Conv2D(in_channels, 256, kernel_size=3, stride=1,padding=1,
                        weight_attr=ParamAttr(initializer=weight_attr))
        self.att = CoordAtt(256)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.conv2_da = nn.Conv2D(256, 1,  kernel_size=3, stride=1,padding=1,
                        weight_attr=ParamAttr(initializer=weight_attr))
        self.avg = nn.AdaptiveAvgPool2D(1)
        
    def forward(self, x):
        x = self.leaky_relu(self.conv1_da(x))
        x = self.att(x)
        x = self.leaky_relu(self.conv2_da(x))
        return self.avg(x)   # n * 1 * 1 * 1
class h_swish(nn.Layer):
    def __init__(self):
        super(h_swish,self).__init__()
        self.relu = nn.ReLU6()
    def forward(self,x):
        return x * self.relu(x+3) / 6
class CoordAtt(nn.Layer):
    def __init__(self, input,reduction=32):
        super(CoordAtt,self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2D((None,1))
        self.pool_w = nn.AdaptiveAvgPool2D((1,None))
        mip = max(8, input // reduction)
        self.conv1 = nn.Conv2D(input, mip, kernel_size=1, stride=1, padding=0,
                            weight_attr=ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()),
                            bias_attr=False)
        self.bn1 = nn.BatchNorm(num_channels = mip,
                        param_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)),
                        bias_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1e-4)),
                        act='relu')
        self.act = h_swish()
        
        self.conv_h = nn.Conv2D(mip, input, kernel_size=1, stride=1, padding=0,
                            weight_attr=ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()),
                            bias_attr=False)
        self.conv_w = nn.Conv2D(mip, input, kernel_size=1, stride=1, padding=0,
                            weight_attr=ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()),
                            bias_attr=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        identity = x
        
        n,c,h,w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        x_w = paddle.transpose(x_w, perm=[0, 1, 3, 2])
 
        y = paddle.concat([x_h, x_w], axis=2)
        y = self.act(self.bn1(self.conv1(y)))
        
        x_h, x_w = paddle.split(y, [h, w], axis=2)
        x_w = paddle.transpose(x_w, perm=[0, 1, 3, 2])
 
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
 
        out = identity * a_w * a_h
 
        return out

class Global_DomainAdaptionModule(nn.Layer):
    def __init__(self):
        super(Global_DomainAdaptionModule,self).__init__()
        self.imgHead = DAImgHead2(2048)
        
    def forward(self, in_features):
        img_grl_fea = gradient_scalar(in_features,-1.0)
        da_img_features = self.imgHead(img_grl_fea)
        da_img_features = F.sigmoid(da_img_features)
        return da_img_features


class DBFPN_domain(nn.Layer):
    def __init__(self, in_channels, out_channels, use_asf=False, **kwargs):
        super(DBFPN_domain, self).__init__()
        self.out_channels = out_channels  # 256
        self.use_asf = use_asf
        weight_attr = paddle.nn.initializer.KaimingUniform()
        # in_channels [256,512,1024,2048]
        self.in2_conv = nn.Conv2D(
            in_channels=in_channels[0],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in3_conv = nn.Conv2D(
            in_channels=in_channels[1],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in4_conv = nn.Conv2D(
            in_channels=in_channels[2],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in5_conv = nn.Conv2D(
            in_channels=in_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p5_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p4_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p3_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p2_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        
        self.Global_DAmodule = Global_DomainAdaptionModule()   # add module 2023.3.15

        if self.use_asf is True:
            self.asf = ASFBlock(self.out_channels, self.out_channels // 4)

    def forward(self, x):
        c2, c3, c4, c5 = x   # 4*256*160*160  4*512*80*80  4*1024*40*40  4*2048*20*20

        in5 = self.in5_conv(c5)  # in5  4*256*20*20

        global_domain_cls = self.Global_DAmodule(c5)

        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1)  # 1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1)  # 1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1)  # 1/4

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)
        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)

        if self.use_asf is True:
            fuse = self.asf(fuse, [p5, p4, p3, p2])  # 256*160*160

        # return fuse
        return {'fuse': fuse,'global_domain_cls':global_domain_cls}


class ASFBlock(nn.Layer):
    """
    This code is refered from:
        https://github.com/MhLiao/DB/blob/master/decoders/feature_attention.py
    """

    def __init__(self, in_channels, inter_channels, out_features_num=4):
        """
        Adaptive Scale Fusion (ASF) block of DBNet++
        Args:
            in_channels: the number of channels in the input data
            inter_channels: the number of middle channels
            out_features_num: the number of fused stages
        """
        super(ASFBlock, self).__init__()
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2D(in_channels, inter_channels, 3, padding=1)

        self.spatial_scale = nn.Sequential(
            #Nx1xHxW
            nn.Conv2D(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                bias_attr=False,
                padding=1,
                weight_attr=ParamAttr(initializer=weight_attr)),
            nn.ReLU(),
            nn.Conv2D(
                in_channels=1,
                out_channels=1,
                kernel_size=1,
                bias_attr=False,
                weight_attr=ParamAttr(initializer=weight_attr)),
            nn.Sigmoid())

        self.channel_scale = nn.Sequential(
            nn.Conv2D(
                in_channels=inter_channels,
                out_channels=out_features_num,
                kernel_size=1,
                bias_attr=False,
                weight_attr=ParamAttr(initializer=weight_attr)),
            nn.Sigmoid())

    def forward(self, fuse_features, features_list):
        fuse_features = self.conv(fuse_features)
        spatial_x = paddle.mean(fuse_features, axis=1, keepdim=True)
        attention_scores = self.spatial_scale(spatial_x) + fuse_features
        attention_scores = self.channel_scale(attention_scores)
        assert len(features_list) == self.out_features_num

        out_list = []
        for i in range(self.out_features_num):
            out_list.append(attention_scores[:, i:i + 1] * features_list[i])
        return paddle.concat(out_list, axis=1)


# model  = Global_DomainAdaptionModule()
# input = paddle.randn([4,2048,20,20])

# momentum = paddle.optimizer.Momentum(learning_rate=0.1, parameters=model.parameters(), weight_decay=0.01)
# import paddle.nn.functional as F
# for _ in range(10):
#     output = model(input)
#     label = paddle.ones_like(output)
#     loss = F.binary_cross_entropy(output,label)
#     print(loss)
#     loss.backward()
#     momentum.step()
#     momentum.clear_grad()
# output = model(input)
# print(output)