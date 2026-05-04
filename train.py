import os
import sys
import argparse
import random
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dataloaders(data_dir, batch_size, val_split=0.2):
    t_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    v_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_ds = datasets.ImageFolder(data_dir, transform=t_transforms)
    class_names = full_ds.classes
    n_samples = len(full_ds)

    indices = list(range(n_samples))
    random.shuffle(indices)
    split_pos = int(n_samples * (1 - val_split))
    
    train_idx = indices[:split_pos]
    val_idx = indices[split_pos:]

    train_data = Subset(full_ds, train_idx)
    val_data = Subset(datasets.ImageFolder(data_dir, transform=v_transforms), val_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print("Loaded dataset from:", data_dir)
    print("Total images =", n_samples)
    print("Train Size:", len(train_idx))
    print("Val Size:", len(val_idx))
    print("Classes mapping:", class_names)
    
    return train_loader, val_loader, class_names

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/tomato")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", default="checkpoints")
    args = parser.parse_args()

    set_seed(42)

    base = os.path.dirname(__file__)
    data_path = os.path.join(base, args.data_dir)
    out_dir = os.path.join(base, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print("Data directory not found:", data_path)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, classes = create_dataloaders(data_path, args.batch_size)
    num_classes = len(classes)

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    
    # Freeze backbone initially
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # unfreeze backbone after epoch 1
        if epoch == 2:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch + 1)
            print("Unfroze backbone!")

        # Train 1 epoch
        model.train()
        t_loss = 0.0
        t_corr = 0
        t_tot = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outs = model(images)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()

            t_loss += loss.item() * images.size(0)
            _, pred = outs.max(1)
            t_corr += pred.eq(labels).sum().item()
            t_tot += labels.size(0)
            
        train_loss = t_loss / t_tot
        train_acc = t_corr / t_tot

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} - Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(out_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": classes,
                "num_classes": num_classes,
            }, best_path)
            print("Saved best model to", best_path)

    final_loc = os.path.join(out_dir, "final_model.pth")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_acc": val_acc,
        "val_loss": val_loss,
        "class_names": classes,
        "num_classes": num_classes,
    }, final_loc)

    cmap_path = os.path.join(out_dir, "class_names.json")
    with open(cmap_path, "w") as f:
        json.dump(classes, f, indent=4)

    # Plot curves
    epochs_range = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train")
    plt.plot(epochs_range, history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], label="Train")
    plt.plot(epochs_range, history["val_acc"], label="Val")
    plt.title("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"))
    plt.close()

    print("Training finished! Best accuracy:", best_acc)

if __name__ == "__main__":
    main()
