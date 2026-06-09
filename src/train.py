from pathlib import Path

import torch
import torch.nn as nn
import wandb
from utils import initialize_wandb, load_checkpoint, save_checkpoint, get_val_dir, get_train_and_val_loader
from config import config
from tqdm.auto import tqdm
from model import VGG19


def train_model(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device,
    resume: bool = True
):
    """
    Trains VGG19 on Tiny ImageNet with:

    - AMP Mixed Precision
    - WandB Logging
    - Early Stopping
    - Resume Training
    - Best Model Checkpointing
    - Periodic Checkpointing
    """

    initialize_wandb(model = model, device = device)

    model = model.to(device)

    wandb.watch(
        model,
        log="all",
        log_freq=100
    )

    print(f"Traing the Model on device = [{device}]")

    scaler = torch.amp.GradScaler()

    loss_fn = nn.CrossEntropyLoss()

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    latest_checkpoint_path = checkpoint_dir / "latest_checkpoint.pth"
    best_checkpoint_path = checkpoint_dir / "best_model.pth"
    best_checkpoint_path = checkpoint_dir / "best_model.pth"

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    # Resume Training
    if resume and latest_checkpoint_path.exists():
        print(
            f"\nLoading Checkpoint: {latest_checkpoint_path}\n"
        )
        
        model, optimizer, scheduler, start_epoch, best_val_loss, _ = load_checkpoint(
            filepath="./checkpoints/latest_checkpoint.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )

        start_epoch += 1

        print(
            f"Resumed from Epoch = {start_epoch}"
        )

    # Epoch Loop
    for epoch in range(start_epoch, config["epochs"]):

        model.train()

        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{config['epochs']}]",
            leave=False
        )

        for images, labels in train_bar:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type = device.type):
                logits = model(images)

                loss = loss_fn(
                    logits,
                    labels
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(
                logits,
                dim=-1
            )

            correct = (
                preds == labels
            ).sum().item()

            batch_acc = correct / labels.size(0)

            running_correct += correct
            running_samples += labels.size(0)

            running_loss += (
                loss.item() * labels.size(0)
            )
            preview_size = min(5, labels.size(0))
            preview = [
                f"{p}->{t}{'  [CORRECT]' if p == t else '  [INCORRECT]'}"
                for p, t in zip(
                    preds[:preview_size].cpu().tolist(),
                    labels[:preview_size].cpu().tolist()
                )
            ]

            train_bar.set_postfix(
                loss=f"{loss.item():.3f}",
                acc=f"{batch_acc*100:.1f}%",
                preds=" | ".join(preview)
            )

        train_loss = running_loss / running_samples
        train_acc = running_correct / running_samples

        # Validation
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type):
                    logits = model(images)
                    loss = loss_fn(
                        logits,
                        labels
                    )
                preds = torch.argmax(
                    logits,
                    dim=-1
                )
                val_correct += (
                    preds == labels
                ).sum().item()
                val_samples += labels.size(0)
                val_loss += (
                    loss.item() * labels.size(0)
                )

        val_loss /= val_samples
        val_acc = val_correct / val_samples
        scheduler.step(val_loss)

        # Terminal Logging
        print(
            f"\nEpoch [{epoch+1}/{config['epochs']}]"
            f" | Train Loss = {train_loss:.4f}"
            f" | Train Acc = {train_acc*100:.2f}%"
            f" | Val Loss = {val_loss:.4f}"
            f" | Val Acc = {val_acc*100:.2f}%"
        )

        # WandB Logging
        wandb.log(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"]
            }
        )

        # Save Latest Checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler = scheduler,
            epoch=epoch,
            loss=val_loss,
            filepath=str(latest_checkpoint_path),
            config = config
        )

        # Save Best Model
        if val_loss < best_val_loss:
            print(
                f"Validation Loss Improved "
                f"({best_val_loss:.4f} -> {val_loss:.4f})"
            )

            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler = scheduler,
                epoch=epoch,
                loss=val_loss,
                filepath=str(best_checkpoint_path),
                config = config
            )

        else:
            patience_counter += 1

        # Periodic Checkpoints
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler = scheduler,
                epoch=epoch,
                loss=val_loss,
                filepath=str(
                    checkpoint_dir /
                    f"epoch_{epoch+1:03d}.pth"
                ),
                config = config
            )

        # Early Stopping
        if patience_counter >= 10:
            print(
                "\nEarly Stopping Triggered\n"
            )
            break

    print(
        "\nTraining Completed Successfully.\n"
    )

    return model, best_val_loss


def main():
    model = VGG19(
        in_channels = config["in_channels"]
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr = config["lr"],
        momentum = config["momentum"],
        weight_decay=config["weight_decay"]
    )
    # Accordingly to the paper: The learning rate was initially set to 1e−2, and then decreased by a factor of 10 when the validation
    # set accuracy stopped improving. In total, the learning rate was decreased 3 times
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3
    )

    # get_val_dir()
    train_loader, val_loader = get_train_and_val_loader()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, best_val_loss = train_model(
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        train_loader = train_loader,
        val_loader = val_loader,
        device = device,
        resume = True
    )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler = scheduler,
        epoch=config["epochs"],
        loss = best_val_loss,
        filepath = "final_model.pth"
    )

if __name__ == "__main__":
    main()