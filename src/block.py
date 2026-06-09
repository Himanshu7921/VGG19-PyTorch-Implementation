import torch
import torch.nn as nn
from config import config

class VGGBlock(nn.Module):
    """
    Implements a reusable VGG building block.

    I wanted the VGG implementation to be modular so that different
    VGG variants can be created without repeating code. After studying
    the original architecture, I noticed that each stage follows the
    same pattern:

        Conv2D -> ReLU -> ... -> MaxPool

    The only thing that changes between stages is the number of
    convolution layers. This block abstracts that pattern and allows
    a stage to be created by simply specifying the number of
    convolution layers required.

    This keeps the implementation cleaner, easier to maintain,
    and closer to the design of the original paper.

    Paper:
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    https://arxiv.org/pdf/1409.1556
    """
    def __init__(self,
                in_channels: int,
                hidden_channels: int,
                conv_kernel_size: int = config["conv_kernel_size"],
                conv_stride: int = config["conv_stride"],
                padding: int = config["padding"],
                debug: bool = False):
                
                super(VGGBlock, self).__init__()

                self.debug = debug

                self.conv_1 = nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = hidden_channels,
                    kernel_size = conv_kernel_size,
                    stride = conv_stride,
                    padding = padding
                )
                self.batch_norm_1 = nn.BatchNorm2d(hidden_channels)
                self.batch_norm_2 = nn.BatchNorm2d(hidden_channels)

                self.relu = nn.ReLU()

                self.conv_2 = nn.Conv2d(
                    in_channels = hidden_channels,
                    out_channels = hidden_channels,
                    kernel_size = conv_kernel_size,
                    stride = conv_stride,
                    padding = padding
                )

                self.block = nn.Sequential(
                    self.conv_1, self.batch_norm_1, self.relu, self.conv_2, self.batch_norm_2, self.relu
                )
    
    def forward(self, x: torch.Tensor):
        if self.debug:
            for i, layer in enumerate(self.block):
                x = layer(x)
                print(f"Layer {i:02d} ({layer.__class__.__name__}): {x.shape}")
            return x

        return self.block(x)
            