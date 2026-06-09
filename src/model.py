import torch.nn as nn
from config import config
from convnets import ConvNets
from fullyconnected import FullyConnectedLayer

class VGG19(nn.Module):
    """
    This class implements the VGG-19 architecture proposed in the paper
    "Very Deep Convolutional Networks for Large-Scale Image Recognition".

    The model consists of a convolutional feature extractor built from
    stacked 3×3 convolution layers and max-pooling operations, followed
    by a fully connected classifier. The implementation is organized into
    separate feature extraction and classification modules to promote
    modularity, reproducibility, and ease of experimentation.

    Proper weight initialization, used by the Original Paper, they explicitly mentioned this:
        - The weights were initialized by sampling from a normal distribution with mean 0 and standard deviation 0.01.
        - The biases were initialized with zero.
    
    > NOTE: I've used Kaiming Normal initialization instead of the paper's initialization strategy,
            as the latter resulted in poor convergence in my experiments.
    
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
                in_features: int = config["fc_in_features"],
                hidden_dim: int = config["fc_hidden_dim"],
                out_features: int = config["n_classes"],
                debug: bool = False):
        super().__init__()


        self.features = ConvNets(
            in_channels = in_channels,
            conv_kernel_size = conv_kernel_size,
            conv_stride = conv_stride,
            padding = padding,
            max_pool_stride = max_pool_stride,
            max_pool_kernel_size = max_pool_kernel_size,
            debug = debug
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.classifier = FullyConnectedLayer(
            in_features = in_features,
            hidden_dim = hidden_dim,
            out_features = out_features
        )
        # self.initialize_weights()
        self.init_weights()
    
    def forward(self, x):
        x = self.features(x)
        assert x.shape[1:] == (512, 7, 7), f"Expected (512, 7, 7), got {x.shape}"
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight,
                    nonlinearity="relu"
                )
                nn.init.zeros_(m.bias)