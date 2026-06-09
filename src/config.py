config = {
    # ------------------------------- ConvNets Configurations -------------------------------
    "conv_kernel_size": 3,
    "conv_stride": 1,
    "padding": 1,
    "max_pool_stride": 2,
    "max_pool_kernel_size": 2,
    "hidden_channel_layer_1": 64,
    # ------------------------------- Fully Connected Layer Configurations -------------------------------
    "fc_in_features": 25088, # Output tensor of 16 ConvNets  = [B, 512, 7, 7] | flattening = 512 * 7 * 7 = 25088
    "fc_hidden_dim": 4096, 
    "n_classes": 200,

    # ------------------------------- VGG-19 Model Specific Configurations -------------------------------
    "in_channels": 3,
    "optimizer": "SGD", # (According to Original VGG-19 Paper)
    "weight_decay": 5e-4,
    "lr": 0.001,
    "momentum": 0.9,
    "batch_size": 24,
    "epochs": 200, # I have early Stopping configured

    # ------------------------------ Image Configurations -------------------------------
    "img_size": 224,
}