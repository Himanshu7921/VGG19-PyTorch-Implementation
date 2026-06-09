import torch
from model import VGG19
from config import config

def test_tensor_shapes():
    vgg_19 = VGG19(
        in_channels = 3,
        debug = True
    )
    H = config["img_size"]
    W = config["img_size"]
    x = torch.rand(1, 3, H, W)
    print(f"Input Tensor Shape = {x.shape}")
    print("-" * 60)
    y = vgg_19(x)
    print("-" * 60)
    print(f"Output Tensor Shape = {y.shape}")

if __name__ == "__main__":
    test_tensor_shapes()