import torch
import torch.nn as nn
from model import VGG19
from config import config
from utils import get_train_and_val_loader, get_val_dir

def overfit(overfit_epoch: int, overfit_batch_size: int = 2):
    model = VGG19(
        in_channels = config["in_channels"]
    )
    overfit_lr = 0.0001
    overfit_momemtum = 0.9
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr= overfit_lr,
        momentum = overfit_momemtum
    )

    def get_accuracy(logits, target):
        y_preds = torch.argmax(logits, dim = -1)
        acc = (y_preds == target).sum() / target.shape[0]
        return acc

    get_val_dir() # remove this if val dir is already organized before; maybe if you ran train.py, then it's already organized
    train_loader, _ = get_train_and_val_loader()

    images, labels = next(iter(train_loader))
    img_overfit = images[:overfit_batch_size]
    labels_overfit = labels[:overfit_batch_size]
    loss_fn = nn.CrossEntropyLoss()

    print(f"Overfitting the Model on Batch Size = {overfit_batch_size} | Epochs = {overfit_epoch}")
    preds = model(img_overfit).argmax(-1)
    print(f"Initial Predictions: {preds}")
    print(f"Original Labels: {labels_overfit}")
    model.train()
    for epoch in range(overfit_epoch):
        logits = model(img_overfit)
        loss = loss_fn(
            logits,
            labels_overfit
        )

        acc = get_accuracy(logits, labels_overfit)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 2 == 0:
            print(f"Epoch [{epoch}/{overfit_epoch}] | Loss = {loss:.3f} | Accuracy = {acc * 100}%")
            if acc.item() >= 0.999:
                print("\n-------------------------------------- Overfitting Sucessfull -------------------------------------- \n")
                print(f"Details:")
                print(f"> Exiting at Epochs: {epoch}")
                print(f"> lr = {overfit_lr}")
                print(" > Forward pass works")
                print(" > Backward pass works")
                print(" > Labels are correct")
                print(" > Optimizer works")
                print(" > Architecture is trainable")
                print(f"\nPredictions = {logits.argmax(-1)} | Original Labels: {labels_overfit}\n")
                break
    return model

if __name__ == "__main__":
    overfit_batch_size = 10
    overfit_epoch = 100

    overfitted_model = overfit(overfit_epoch = overfit_epoch, overfit_batch_size = overfit_batch_size)
    print(f"\nSucessfully Overfitted on {overfit_batch_size} Batch")
    print(f"\nReady to Train on Actual Dataset")