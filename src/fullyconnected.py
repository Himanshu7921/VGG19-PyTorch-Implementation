import torch
import torch.nn as nn
class FullyConnectedLayer(nn.Module):
    """
    This class implements the fully connected classification head of the
    VGG-19 architecture.

    The classifier consists of three linear layers with ReLU activations
    that transform high-level convolutional features into class scores.
    Prior to entering this module, spatial feature maps are flattened into
    a one-dimensional feature vector, enabling image-level classification.

    Paper:
    Very Deep Convolutional Networks for Large-Scale Image Recognition

    Link:
    https://arxiv.org/pdf/1409.1556
    """
    def __init__(self,
                in_features: int,
                hidden_dim: int,
                out_features: int):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features

        self.linear_layer_1 = nn.Linear(in_features = self.in_features, out_features = self.hidden_dim)
        self.linear_layer_2 = nn.Linear(in_features = self.hidden_dim, out_features = self.hidden_dim)
        self.linear_layer_3 = nn.Linear(in_features = self.hidden_dim, out_features = self.out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.5)

        self.fc_layers = nn.Sequential(
            self.linear_layer_1, self.relu, self.dropout, self.linear_layer_2, self.relu, self.dropout, self.linear_layer_3
        )
    
    def forward(self, x: torch.Tensor):
        return self.fc_layers(x)