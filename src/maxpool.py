import torch
import torch.nn as nn
from config import config

class MaxPool2d(nn.Module):
    """
    This class implements the max-pooling operation used throughout the
    VGG-19 architecture.

    Max pooling reduces the spatial resolution of feature maps while
    preserving the most dominant activations within each local region.
    This operation increases the receptive field of subsequent layers
    and improves computational efficiency by progressively downsampling
    the feature representations.

    Paper:
    Very Deep Convolutional Networks for Large-Scale Image Recognition

    Link:
    https://arxiv.org/pdf/1409.1556
    """
    def __init__(self,
            max_pool_stride: int = config["max_pool_stride"],
            max_pool_kernel_size: int = config["max_pool_kernel_size"]):
        
        super(MaxPool2d, self).__init__()
        
        self.max_pool_stride = max_pool_stride
        self.max_pool_kernel_size = max_pool_kernel_size

        self.max_pool2d = nn.MaxPool2d(
            kernel_size = self.max_pool_kernel_size,
            stride = self.max_pool_stride
        )
    
    def forward(self, x: torch.Tensor):
        return self.max_pool2d(x)