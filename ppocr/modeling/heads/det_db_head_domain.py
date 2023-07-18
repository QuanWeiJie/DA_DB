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

import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.autograd import PyLayer

def get_bias_attr(k):
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = paddle.nn.initializer.Uniform(-stdv, stdv)
    bias_attr = ParamAttr(initializer=initializer)
    return bias_attr

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

        self.conv1_da = nn.Conv2D(in_channels, in_channels//4, kernel_size=1, stride=1,
                        weight_attr=ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()),
                        bias_attr=get_bias_attr(in_channels // 4))
        self.conv_bn1 = nn.BatchNorm(
                        num_channels=in_channels // 4,
                        param_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)),
                        bias_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1e-4)),
                        act='relu')
        self.conv2_da = nn.Conv2D(in_channels//4, 1, kernel_size=1, stride=1,
                        weight_attr=ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()),
                        bias_attr=get_bias_attr(in_channels // 4))
        
    def forward(self, x):
        t = self.conv_bn1(self.conv1_da(x))
        return self.conv2_da(t)

class DomainAdaptionModule(nn.Layer):
    def __init__(self):
        super(DomainAdaptionModule,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.imgHead = DAImgHead(256)
        
    def forward(self, in_features):
        img_grl_fea = gradient_scalar(self.avgpool(in_features),-1.0)
        da_img_features = F.sigmoid(self.imgHead(img_grl_fea))
        return da_img_features

class DomainAdaptionModule_new(nn.Layer):
    def __init__(self,in_channels, kernel_list=[3, 2, 2], **kwargs):
        super(DomainAdaptionModule_new,self).__init__()
        self.c1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
            weight_attr=ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=False)
        self.c_bn1 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act='relu')
        self.c2 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4))
        self.c_bn2 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act="relu")
        self.c3 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=kernel_list[2],
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4))
        
    def forward(self, x):
        x = gradient_scalar(x,-1.0)
        x = self.c1(x)
        x = self.c_bn1(x)
        x = self.c2(x)
        x = self.c_bn2(x)
        x = self.c3(x)
        x = F.sigmoid(x)

        return x


class Head(nn.Layer):
    def __init__(self, in_channels, kernel_list=[3, 2, 2], **kwargs):
        super(Head, self).__init__()

        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
            weight_attr=ParamAttr(),
            bias_attr=False)
        self.conv_bn1 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act='relu')
        self.conv2 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4))
        self.conv_bn2 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act="relu")
        self.conv3 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=kernel_list[2],
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4), )
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        x = self.conv3(x)
        x = F.sigmoid(x)
        
        return x


class DBHead_domain(nn.Layer):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50, **kwargs):
        super(DBHead_domain, self).__init__()
        self.k = k
        self.binarize = Head(in_channels, **kwargs)
        self.thresh = Head(in_channels, **kwargs)
        # self.DAmodule = DomainAdaptionModule()   # add module 2023.2.28
        self.DAmodule = DomainAdaptionModule_new(in_channels,**kwargs)   # add module 2023.5.6  size: 640*640

    def step_function(self, x, y):
        return paddle.reciprocal(1 + paddle.exp(-self.k * (x - y)))

    def forward(self, x, targets=None):
        # {'fuse': fuse,'global_domain_cls':global_domain_cls}
        flag = 0
        global_domain_cls = None
        if isinstance(x,dict):
            if 'global_domain_cls' in x:
                global_domain_cls = x['global_domain_cls']
                flag = 1
            x = x['fuse']
        shrink_maps = self.binarize(x)
        # if not self.training:
        #     return {'maps': shrink_maps}

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        if not self.training:   # 
            return {'maps': binary_maps}
        y = paddle.concat([shrink_maps, threshold_maps, binary_maps], axis=1)
        domain_cls = self.DAmodule(x)
        # domain_cls = None
        return {'maps': y,'global_domain_cls':global_domain_cls,'domain_cls':domain_cls}