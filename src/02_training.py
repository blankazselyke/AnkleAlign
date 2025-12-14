import os

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from config import OUTPUT_DIR, MODEL_SAVE_PATH, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, \
    EARLY_STOPPING_PATIENCE, LR_SCHEDULER_FACTOR, LR_SCHEDULER_PATIENCE, CSV_PATH_TRAIN, CSV_PATH_VAL, CSV_PATH_TEST
from utils import setup_logger, count_parameters

logger = setup_logger()


class AugmentedAnkleDataset(Dataset):
    """
    Dataset wrapper that expands the training set by applying multiple
    distinct transformations to each image.

    Structure:
    Each index in the dataset corresponds to a specific (Image, Transform) pair.
    index // N -> Original image index
    index % N  -> Transformation index
    """

    def __init__(self, df, root_dir, transforms_list):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transforms_list = transforms_list
        self.n_views = len(transforms_list)

    def __len__(self):
        return len(self.df) * self.n_views

    def __getitem__(self, idx):
        base_idx = idx // self.n_views
        transform_idx = idx % self.n_views

        row = self.df.iloc[base_idx]
        path = os.path.join(self.root_dir, row["filename"])

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Image load failed ({path}): {e}")
            img = Image.new("RGB", (224, 224))

        img = self.transforms_list[transform_idx](img)
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return img, label


class AnkleDataset(Dataset):
    # --- JAVÍTÁS 2: transform átvétele paraméterként ---
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform  # Elmentjük

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.root_dir, row['filename'])

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Hiba a képnél ({path}): {e}")
            img = Image.new('RGB', (224, 224))

        # --- JAVÍTÁS 3: A self.transform használata ---
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(int(row['label']), dtype=torch.long)


def main():
    logger.info("--- 2. STEP: TRAINING STARTED ---")

    logger.info("Hyperparameters:")
    logger.info(f"EPOCHS: {EPOCHS}")
    logger.info(f"BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"LEARNING_RATE: {LEARNING_RATE}")
    logger.info(f"WEIGHT_DECAY: {WEIGHT_DECAY}")
    logger.info(f"EARLY_STOPPING_PATIENCE: {EARLY_STOPPING_PATIENCE}")
    logger.info(f"LR_SCHEDULER_FACTOR: {LR_SCHEDULER_FACTOR}")
    logger.info(f"LR_SCHEDULER_PATIENCE: {LR_SCHEDULER_PATIENCE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Data Loading
    train_df = pd.read_csv(CSV_PATH_TRAIN)
    val_df = pd.read_csv(CSV_PATH_VAL)
    test_df = pd.read_csv(CSV_PATH_TEST)

    logger.info(f"Data loaded successfully. Total samples: {len(train_df) + len(val_df) + len(test_df)}")

    # 1. Deterministic Center Crop
    center_crop_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 2. Rotation & Color
    aug_transform_1 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 3. Affine & Blur
    aug_transform_2 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.9, 1.0)),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Validation Transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Initialize Datasets
    # Training uses the Augmented dataset (3x expansion)
    train_ds = AugmentedAnkleDataset(
        train_df,
        OUTPUT_DIR,
        transforms_list=[
            center_crop_transform,
            aug_transform_1,
            aug_transform_2
        ]
    )

    val_ds = AnkleDataset(val_df, OUTPUT_DIR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize Model (ResNet18)
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    logger.info("--- ResNet18 Model Architecture ---")
    logger.info(str(model))

    total_params, trainable_params = count_parameters(model)
    non_trainable_params = total_params - trainable_params

    logger.info("--- Parameter Count ---")
    logger.info(f"  Total Parameters:         {total_params}")
    logger.info(f"  Trainable Parameters:     {trainable_params}")
    logger.info(f"  Non-trainable Parameters: {non_trainable_params}")

    # Optimization Setup
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', # We monitor validation loss
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE
    )

    # Training Loop
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        epoch_loss = train_loss / total
        epoch_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                _, pred = torch.max(out, 1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
                loss = criterion(out, y)
                val_loss += loss.item() * X.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        logger.info(f"Epoch {epoch + 1}/{EPOCHS} - "
                    f"Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f} - "
                    f"Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"New best model saved! Val loss: {best_loss:.4f}")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epochs.")

        scheduler.step(avg_val_loss)

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping triggered.")
            break

    logger.info("Training finished.")


if __name__ == "__main__":
    main()
