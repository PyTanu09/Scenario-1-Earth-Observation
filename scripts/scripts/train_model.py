import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

# ----------------------------
# Dataset Class
# ----------------------------
class LandCoverDataset(Dataset):
    def __init__(self, df, images_dir=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row.filename

        img = None
        if self.images_dir:
            img_path = os.path.join(self.images_dir, f"{fname}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")

        # fallback to NPY patch
        if img is None:
            if hasattr(row, "patch_path") and os.path.exists(row.patch_path):
                arr = np.load(row.patch_path)
                arr = arr.astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
                img = Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")
            else:
                raise RuntimeError(f"Image not found for {fname}")

        if self.transform:
            img = self.transform(img)

        label = label2idx[row.label]
        return img, label, fname

# ----------------------------
# Main Script
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="outputs/datasets/labels.csv")
    parser.add_argument("--images_dir", default="data/rgb")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    LABELS_CSV = args.labels
    IMAGES_DIR = args.images_dir
    OUT_DIR = args.out_dir
    MODEL_DIR = os.path.join(OUT_DIR, "models")
    FIG_DIR = os.path.join(OUT_DIR, "figures")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # ----------------------------
    # Load labels CSV
    # ----------------------------
    df = pd.read_csv(LABELS_CSV)
    train_df = df[df.split == "train"].reset_index(drop=True)
    test_df = df[df.split == "test"].reset_index(drop=True)

    labels = sorted(df.label.unique())
    label2idx = {l: i for i, l in enumerate(labels)}
    idx2label = {i: l for l, i in label2idx.items()}

    # ----------------------------
    # Transforms
    # ----------------------------
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # ----------------------------
    # Datasets & Dataloaders
    # ----------------------------
    train_ds = LandCoverDataset(train_df, images_dir=IMAGES_DIR, transform=train_transform)
    test_ds = LandCoverDataset(test_df, images_dir=IMAGES_DIR, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ----------------------------
    # Model
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(labels)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ----------------------------
    # Training Loop
    # ----------------------------
    best_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for imgs, labs, _ in train_loader:
            imgs = imgs.to(device)
            labs = labs.to(device)

            out = model(imgs)
            loss = criterion(out, labs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # ----------------------------
        # Evaluation
        # ----------------------------
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labs, _ in test_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labs.numpy())

        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss:.4f} | Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "resnet18_best.pth"))

    # ----------------------------
    # Confusion Matrix Plot
    # ----------------------------
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(range(num_classes), [idx2label[i] for i in range(num_classes)], rotation=45)
    plt.yticks(range(num_classes), [idx2label[i] for i in range(num_classes)])
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrix.png"), dpi=300)

    # ----------------------------
    # Correct vs Incorrect Samples
    # ----------------------------
    correct_idx = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p == l][:5]
    incorrect_idx = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p != l][:5]

    test_imgs = []
    test_labels_list = []
    for img, lab, _ in test_ds:
        test_imgs.append(img)
        test_labels_list.append(lab)

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, ci in enumerate(correct_idx):
        img = test_imgs[ci].permute(1, 2, 0).cpu().numpy()
        img = (img * imagenet_std + imagenet_mean).clip(0,1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"GT:{idx2label[test_labels_list[ci]]}\nPred:{idx2label[all_preds[ci]]}")
        axes[0, i].axis('off')

    for i, ci in enumerate(incorrect_idx):
        img = test_imgs[ci].permute(1, 2, 0).cpu().numpy()
        img = (img * imagenet_std + imagenet_mean).clip(0,1)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"GT:{idx2label[test_labels_list[ci]]}\nPred:{idx2label[all_preds[ci]]}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pred_examples.png"), dpi=300)

    print("✓ Training & evaluation complete!")
    print("✓ Figures saved in:", FIG_DIR)
    print("✓ Best model saved in:", MODEL_DIR)









