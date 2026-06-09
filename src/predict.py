import torch
from utils import load_checkpoint, get_class_names, visualize_predictions
import torch
from config import config
from model import VGG19

def get_model(filepath, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG19(
        in_channels = config["in_channels"]
    )
    model, _, _, _, _, config = load_checkpoint(
        filepath = filepath,
        model=model,
        optimizer=None,
        scheduler=None,
        device=device
    )
    model.eval()
    return model

def main():
    model = get_model(filepath="./checkpoints/best_model.pth", config = config)
    word_file_path = "./data/tiny-imagenet-200/words.txt"
    train_folder_path = "./data/tiny-imagenet-200/train"
    class_names = get_class_names(word_file_path, train_folder_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    visualize_predictions(
        test_dir="./data/tiny-imagenet-200/test/images",
        model=model,
        idx_to_class_name=class_names,
        device=device
    )

if __name__ == "__main__":
    main()