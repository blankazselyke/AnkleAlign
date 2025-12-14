import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import CSV_PATH_TRAIN, CSV_PATH_VAL, OUTPUT_DIR, EPOCHS_BASE, BATCH_SIZE_BASE, LEARNING_RATE_BASE, \
    BASELINE_SAVE_PATH
from utils import setup_logger, count_parameters

logger = setup_logger()


class CNN(nn.Module):
    """
    A lightweight CNN trained from scratch.
    Serves as a baseline to compare against ResNet18.
    """
    def __init__(self, num_classes=3):
        super(CNN, self).__init__()

        # Block 1 (Input: 3 x 224 x 224)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        # Output after Conv: 8 x 224 x 224 -> After Pool: 8 x 112 x 112

        # Block 2
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        # Output after Conv: 16 x 112 x 112 -> After Pool: 16 x 56 x 56

        # Block 3
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Output after Conv: 32 x 56 x 56 -> After Pool: 32 x 28 x 28

        # Shared pooling layer (used by all three blocks)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected Layers (Input size: 32 * 28 * 28 = 25088)
        self.fc1 = nn.Linear(32 * 28 * 28, 128)  # First FC layer
        self.fc2 = nn.Linear(128, num_classes)  # Second (output) FC layer

    def forward(self, x):
        # Convolutional steps
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten
        x = x.view(-1, 32 * 28 * 28)

        # Fully Connected steps
        x = F.relu(self.fc1(x))  # Activation function needed after the first FC layer
        x = self.fc2(x)  # No activation needed at output (handled by CrossEntropyLoss)

        return x


class BaselineDataset(Dataset):
    """
    Simple dataset loader for the baseline model.
    """

    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])

        try:
            img = Image.open(img_path).convert("RGB")
        except (OSError, FileNotFoundError):
            # Return black image on failure to prevent crash
            img = Image.new('RGB', (224, 224))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(int(row['label']))

def main():
    logger.info("--- BASELINE MODEL TRAINING STARTED ---")

    logger.info("Baseline Hyperparameters:")
    logger.info(f"  EPOCHS: {EPOCHS_BASE}")
    logger.info(f"  BATCH_SIZE: {BATCH_SIZE_BASE}")
    logger.info(f"  LEARNING_RATE: {LEARNING_RATE_BASE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Data Loading
    train_df = pd.read_csv(CSV_PATH_TRAIN)
    val_df = pd.read_csv(CSV_PATH_VAL)

    # Standard Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_loader = DataLoader(
        BaselineDataset(train_df, OUTPUT_DIR, transform),
        batch_size=BATCH_SIZE_BASE,
        shuffle=True
    )
    val_loader = DataLoader(
        BaselineDataset(val_df, OUTPUT_DIR, transform),
        batch_size=BATCH_SIZE_BASE,
        shuffle=False
    )

    # Model Initialization
    model = CNN(num_classes=3).to(device)

    logger.info("--- Baseline Model Architecture ---")
    logger.info(str(model))

    total_params, trainable_params = count_parameters(model)
    non_trainable_params = total_params - trainable_params

    logger.info("--- Parameter Count ---")
    logger.info(f"  Total Parameters:         {total_params}")
    logger.info(f"  Trainable Parameters:     {trainable_params}")
    logger.info(f"  Non-trainable Parameters: {non_trainable_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_BASE)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    logger.info("Starting baseline training...")
    best_acc = 0.0

    for epoch in range(EPOCHS_BASE):
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

        logger.info(f"Epoch {epoch + 1}/{EPOCHS_BASE} - "
                    f"Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f} - "
                    f"Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), BASELINE_SAVE_PATH)

    logger.info(f"Baseline training finished.")


if __name__ == "__main__":
    main()