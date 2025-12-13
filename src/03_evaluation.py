import torch
import torch.nn as nn
import torch.nn.functional as F  # Needed for the CNN class
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image, ImageOps
import os
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from config import CSV_PATH_TEST, OUTPUT_DIR, MODEL_SAVE_PATH, BASELINE_SAVE_PATH, BATCH_SIZE
from utils import setup_logger

logger = setup_logger()


# --- 1. Define Baseline Architecture (Must match 02-baseline.py) ---
class CNN(nn.Module):
    """
    The exact same architecture used in 02-baseline.py.
    """

    def __init__(self, num_classes=3):
        super(CNN, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        # Block 2
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        # Block 3
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Shared Pooling
        self.pool = nn.MaxPool2d(2, 2)
        # FC Layers
        self.fc1 = nn.Linear(32 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- 2. Test Dataset Class ---
class TestDataset(Dataset):
    """Dataset for loading test images with a specific transform."""

    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(OUTPUT_DIR, row["filename"])

        try:
            img = Image.open(img_path).convert("RGB")
            # Fix EXIF rotation if needed
            img = ImageOps.exif_transpose(img)
        except Exception as e:
            # logger.warning(f"Image load error ({img_path}): {e}")
            img = Image.new("RGB", (224, 224))

        img = self.transform(img)
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return img, label


# --- 3. Evaluation Helper Function ---
def evaluate_model(model, model_name, test_loader, device):
    logger.info(f"--- Evaluating: {model_name} ---")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    accuracy = (all_preds == all_labels).mean()
    logger.info(f"FINAL TEST ACCURACY ({model_name}): {accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"Confusion Matrix ({model_name}):")
    logger.info(f"\n{cm}")

    # Classification Report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=["Pronation", "Neutral", "Supination"],
        zero_division=0
    )
    logger.info(f"Classification Report ({model_name}):")
    logger.info(f"\n{report}")
    logger.info("-" * 40)


def main():
    logger.info("--- 3. STEP: EVALUATION STARTED ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Check that CSV exists
    if not CSV_PATH_TEST.exists():
        logger.error(f"Missing test CSV file: {CSV_PATH_TEST}")
        return

    df = pd.read_csv(CSV_PATH_TEST)
    logger.info(f"Test samples loaded: {len(df)}")

    # ==========================================
    # 1. EVALUATE BASELINE MODEL
    # ==========================================
    if BASELINE_SAVE_PATH.exists():
        # Baseline uses simple normalization (0.5)
        baseline_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        baseline_loader = DataLoader(TestDataset(df, baseline_transform), batch_size=BATCH_SIZE, shuffle=False)

        # Init and Load Baseline
        baseline_model = CNN(num_classes=3).to(device)
        baseline_model.load_state_dict(torch.load(BASELINE_SAVE_PATH, map_location=device))

        evaluate_model(baseline_model, "Baseline (CNN)", baseline_loader, device)
    else:
        logger.warning(f"Baseline model not found at {BASELINE_SAVE_PATH}. Skipping.")

    # ==========================================
    # 2. EVALUATE MAIN MODEL (ResNet)
    # ==========================================
    if MODEL_SAVE_PATH.exists():
        # Main model uses ImageNet normalization
        main_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        main_loader = DataLoader(TestDataset(df, main_transform), batch_size=BATCH_SIZE, shuffle=False)

        # Init and Load ResNet
        main_model = models.resnet18(weights=None)

        # NOTE: If you used the Dropout architecture in training, use this:
        # main_model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 3))

        # If you used the standard linear layer, use this:
        main_model.fc = nn.Linear(main_model.fc.in_features, 3)

        main_model = main_model.to(device)

        try:
            main_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            evaluate_model(main_model, "Main Model (ResNet18)", main_loader, device)
        except RuntimeError as e:
            logger.error(f"Failed to load Main Model weights: {e}")
            logger.error(
                "Did you change the architecture (e.g. added Dropout) in training? Update evaluation.py to match!")
    else:
        logger.warning(f"Main model not found at {MODEL_SAVE_PATH}. Skipping.")

    logger.info("--- EVALUATION FINISHED SUCCESSFULLY ---")


if __name__ == "__main__":
    main()
