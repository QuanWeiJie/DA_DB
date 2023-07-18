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




class EASTHead_domain(nn.Layer):
    """
    """
    def __init__(self, in_channels, model_name, **kwargs):
        super(EASTHead_domain, self).__init__()  # python2 
        self.model_name = model_name
        if self.model_name == "large":
            num_outputs = [128, 64, 1, 8]
        else:
            num_outputs = [64, 32, 1, 8]

        self.det_conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=num_outputs[0],
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu')
        self.det_conv2 = ConvBNLayer(
            in_channels=num_outputs[0],
            out_channels=num_outputs[1],
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu')
        self.score_conv = ConvBNLayer(
            in_channels=num_outputs[1],
            out_channels=num_outputs[2],
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)
        
        self.geo_conv = ConvBNLayer(
            in_channels=num_outputs[1],
            out_channels=num_outputs[3],
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)
        

        self.da_det_conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=num_outputs[0],
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu')
        self.da_det_conv2 = ConvBNLayer(
            in_channels=num_outputs[0],
            out_channels=num_outputs[1],
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu')
        self.da_score_conv = ConvBNLayer(
            in_channels=num_outputs[1],
            out_channels=num_outputs[2],
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)

    def forward(self, x, targets=None):
        if isinstance(x,dict):
            global_domain_cls = x['global_domain_cls']
            x = x['g']
        f_det = self.det_conv1(x)
        f_det = self.det_conv2(f_det)
        f_score = self.score_conv(f_det)  # 8*1*128*128

        # add
        xx  = gradient_scalar(x,-1.0)
        domain_cls = F.sigmoid(self.da_score_conv(self.da_det_conv2(self.da_det_conv1(xx))))
        # add end

        f_score = F.sigmoid(f_score)
        f_geo = self.geo_conv(f_det)
        f_geo = (F.sigmoid(f_geo) - 0.5) * 2 * 800

        pred = {'global_domain_cls':global_domain_cls,'domain_cls': domain_cls, 'f_score': f_score, 'f_geo': f_geo}
        return pred
