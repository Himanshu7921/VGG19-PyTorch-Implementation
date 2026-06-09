import torch
import torch.nn as nn
from config import config
from block import VGGBlock
from maxpool import MaxPool2d

class ConvNets(nn.Module):
    """
    This class implements the convolutional feature extraction backbone of
    the VGG-19 architecture.

    The backbone is composed of stacked VGG blocks and max-pooling layers
    that progressively transform the input image into high-level feature
    representations. Spatial resolution is reduced through pooling while
    the number of feature channels is increased, enabling hierarchical
    feature learning across multiple scales.

    Paper:
    Very Deep Convolutional Networks for Large-Scale Image Recognition

    Link:
    https://arxiv.org/pdf/1409.1556
    """
    def __init__(self,
                in_channels: int,
                conv_kernel_size: int = config["conv_kernel_size"],
                conv_stride: int = config["conv_stride"],
                padding: int = config["padding"],
                max_pool_stride: int = config["max_pool_stride"],
                max_pool_kernel_size: int = config["max_pool_kernel_size"],
                debug: bool = False):
        super(ConvNets, self).__init__()
        self.in_channels = in_channels
        prev_channels = config["hidden_channel_layer_1"]

        # Inserting 1st Layer (3 channel -> 64 channels) + MaxPool [2 ConvNets + ReLU]
        self.vgg_19_blocks = [
            nn.Sequential(VGGBlock(
                    in_channels = self.in_channels,
                    hidden_channels = config["hidden_channel_layer_1"],
                    conv_kernel_size = conv_kernel_size,
                    conv_stride = conv_stride,
                    padding = padding,
                    debug = debug
            ), MaxPool2d(
                max_pool_stride = max_pool_stride,
                max_pool_kernel_size = max_pool_kernel_size
            ))
        ]
 
        # Inserting 2nd Layer (64 channel -> 128 channels) + MaxPool [2 ConvNets + ReLU]
        self.vgg_19_blocks.append(
            nn.Sequential(VGGBlock(
                in_channels = prev_channels,
                hidden_channels = 2 * prev_channels,
                conv_kernel_size = conv_kernel_size,
                conv_stride = conv_stride,
                padding = padding,
                debug = debug
            ),
            MaxPool2d(
                max_pool_stride = max_pool_stride,
                max_pool_kernel_size = max_pool_kernel_size
            ))
        )
        prev_channels *= 2

        # Inserting remaining 8 ConvNets + 2 MaxPools
        for _ in range(2):
            self.vgg_19_blocks.append(
                nn.Sequential(
                # 2 set of ConvNets with ReLU
                VGGBlock(
                    in_channels = prev_channels,
                    hidden_channels = 2 * prev_channels,
                    conv_kernel_size = conv_kernel_size,
                    conv_stride = conv_stride,
                    padding = padding,
                    debug = debug
                ),
                # Again 2 set of ConvNets with ReLU
                VGGBlock(
                    in_channels = 2 * prev_channels,
                    hidden_channels = 2 * prev_channels,
                    conv_kernel_size = conv_kernel_size,
                    conv_stride = conv_stride,
                    padding = padding,
                    debug = debug
                ),
                # MaxPool at the end of 4 consecutive ConvNets
                MaxPool2d(
                    max_pool_stride = max_pool_stride,
                    max_pool_kernel_size = max_pool_kernel_size
                ))
            )
            prev_channels *= 2

        # Insterting Last set of 4 ConvNets + 1 MaxPool
        self.vgg_19_blocks.append(
            nn.Sequential(
            # 2 set of ConvNets with ReLU
            VGGBlock(
                in_channels = prev_channels,
                hidden_channels = prev_channels,
                conv_kernel_size = conv_kernel_size,
                conv_stride = conv_stride,
                padding = padding,
                debug = debug
            ),
            # Again 2 set of ConvNets with ReLU
            VGGBlock(
                in_channels = prev_channels,
                hidden_channels = prev_channels,
                conv_kernel_size = conv_kernel_size,
                conv_stride = conv_stride,
                padding = padding,
                debug = debug
            ),

            # MaxPool at the end of 4 consecutive ConvNets
            MaxPool2d(
                max_pool_stride = max_pool_stride,
                max_pool_kernel_size = max_pool_kernel_size
            ))
        )
        self.vgg_19_layers = nn.Sequential(*self.vgg_19_blocks)
    
    def forward(self, x: torch.Tensor):
        return self.vgg_19_layers(x)