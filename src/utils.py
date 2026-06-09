import torch
import torch.nn as nn
from config import config
import os
import shutil
from torchvision import datasets
from torchvision import transforms
import wandb
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

def get_params(model):
    feature_params = sum(p.numel() for p in model.features.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    total_params = feature_params + classifier_params
    return feature_params, classifier_params, total_params

def print_params_details(model):
    feature_params, classifier_params, total_params = get_params(model = model)
    print("=" * 60)
    print("Model Parameter Statistics")
    print("=" * 60)
    print(f"Feature Parameters: {feature_params:,} ({feature_params/1e6:.2f} M)")
    print(f"Classifier Parameters: {classifier_params:,} ({classifier_params/1e6:.2f} M)")
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f} M)")
    print("-" * 60)
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f} M)")
    print("=" * 60)
    

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    loss: float,
    filepath: str,
    config: dict
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "config": config
    }

    torch.save(checkpoint, filepath)

def load_checkpoint(
    filepath: str,
    model,
    optimizer=None,
    scheduler = None,
    device="cpu"
):
    checkpoint = torch.load(
        filepath,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        
    if scheduler is not None:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)
    config = checkpoint.get("config")

    return model, optimizer, scheduler, epoch, loss, config

def get_val_dir():
    # Run this Once
    val_dir = "./data/tiny-imagenet-200/val"
    img_dir = os.path.join(val_dir, "images")
    annotation_file = os.path.join(val_dir, "val_annotations.txt")

    with open(annotation_file, "r") as f:
        for line in f.readlines():
            image_name, class_name, *_ = line.split("\t")

            class_dir = os.path.join(val_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            src = os.path.join(img_dir, image_name)
            dst = os.path.join(class_dir, image_name)

            if os.path.exists(src):
                shutil.move(src, dst)

    shutil.rmtree(img_dir)

    print("Validation folder reorganized successfully.")


def get_train_and_val_loader():

    IMAGE_SIZE = config["img_size"]
    BATCH_SIZE = config["batch_size"]
    NUM_WORKERS = 4

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(
            IMAGE_SIZE,
            padding=4
        ),
        transforms.ToTensor(),

        # Normalize accross channel_dim; means = [red_mean, green_mean, blue_mean]; similary for std
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ) # There's numbers are analyzed accross Millions of Images and then they're computed 
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(
        root="./data/tiny-imagenet-200/train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root="./data/tiny-imagenet-200/val",
        transform=val_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Train Samples: {len(train_dataset):,}")
    print(f"Validation Samples: {len(val_dataset):,}")
    print(f"Classes: {len(train_dataset.classes)}")
    return train_loader, val_loader

def load_wordnet_mapping(words_file):
    wnid_to_name = {}

    with open(words_file, "r") as f:
        for line in f:
            wnid, label = line.strip().split("\t")
            wnid_to_name[wnid] = label

    return wnid_to_name

def get_class_names(word_file_path, train_folder_path):
    wnid_to_name = load_wordnet_mapping(
        word_file_path
    )

    train_dataset = datasets.ImageFolder(
        root = train_folder_path,
        transform = None
    )

    class_names = train_dataset.classes
    class_names = [
        wnid_to_name[wnid]
        for wnid in train_dataset.classes
    ]
    return class_names

def visualize_random_samples(
    train_loader,
    class_names=None,
    n_samples: int = 5
):
    """
    Visualize random images and labels from a DataLoader.

    Args:
        train_loader: PyTorch DataLoader
        class_names: Optional mapping from class index -> class name
        n_samples: Number of images to display
    """

    images, labels = next(iter(train_loader))

    indices = random.sample(
        range(len(images)),
        k=min(n_samples, len(images))
    )

    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    fig, axes = plt.subplots(
        1,
        len(indices),
        figsize=(15, 4)
    )

    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):

        image = images[idx]

        # De-normalize
        image = image * std + mean

        image = image.permute(
            1,
            2,
            0
        )

        image = image.clamp(
            0,
            1
        )

        label = labels[idx].item()

        if class_names is not None:
            title = class_names[label]
        else:
            title = f"Class {label}"

        ax.imshow(image.numpy())
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()



def initialize_wandb(device, model):
    feature_params, classifier_params, total_params = get_params(model)
    wandb.init(
        project="vgg19-from-scratch",
        name="vgg19_tinyimagenet_adamw_original_init",
        tags=[
            "vgg19",
            "tiny-imagenet",
            "from-scratch",
            "paper-reproduction"
        ],

        config={
            # Dataset
            "dataset": "TinyImageNet",
            "num_classes": config["n_classes"],
            "image_size": config["img_size"],

            # Architecture
            "architecture": "VGG19",
            "conv_layers": 16,
            "maxpool_layers": 5,
            "fc_layers": 3,

            # Parameters
            "feature_params": feature_params,
            "classifier_params": classifier_params,
            "total_params": total_params,

            # Training
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],

            # Optimizer
            "optimizer": config["optimizer"],
            "learning_rate": config["lr"],
            "momentum": config["momentum"],
            "weight_decay": config["weight_decay"],

            # Initialization
            "weight_init": "Normal(0, 0.01)",
            "bias_init": "Zero",

            # Loss
            "loss_fn": "CrossEntropyLoss",

            # Hardware
            "device": str(device)
        }
    )




def visualize_predictions(
    test_dir,
    model,
    idx_to_class_name,
    device,
    n_samples=5
):
    model.eval()

    image_paths = list(
        Path(test_dir).glob("*.JPEG")
    )

    selected = random.sample(
        image_paths,
        n_samples
    )

    IMAGE_SIZE = config["img_size"]
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    display_transform = transforms.Resize(
        (1024, 1024)
    )
    

    fig, axes = plt.subplots(
        1,
        n_samples,
        figsize=(18, 5)
    )

    if n_samples == 1:
        axes = [axes]

    with torch.no_grad():

        for ax, image_path in zip(
            axes,
            selected
        ):

            pil_image = Image.open(
                image_path
            ).convert("RGB")

            # For model inference
            x = val_transform(
                pil_image
            )

            logits = model(
                x.unsqueeze(0).to(device)
            )

            pred = torch.argmax(
                logits,
                dim=-1
            ).item()

            confidence = (
                torch.softmax(
                    logits,
                    dim=-1
                )[0, pred]
                .item()
            )

            # For visualization only
            display_image = display_transform(
                pil_image
            )

            ax.imshow(display_image)

            ax.set_title(
                f"{idx_to_class_name[pred][:25]}\n"
                f"{confidence:.2%}"
            )

            ax.axis("off")

    plt.tight_layout()
    plt.show()