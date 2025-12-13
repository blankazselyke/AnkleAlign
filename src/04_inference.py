import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
from pathlib import Path
from config import DATA_DIR, OUTPUT_DIR, MODEL_SAVE_PATH
from utils import setup_logger

logger = setup_logger()


def load_model():
    """Load the trained ResNet18 model with the correct output layer."""
    if not MODEL_SAVE_PATH.exists():
        logger.error(f"Missing model file: {MODEL_SAVE_PATH}")
        return None

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)

    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
    model.eval()

    logger.info(f"Model successfully loaded from {MODEL_SAVE_PATH}")
    return model


def load_image(image_path):
    """Load and preprocess a single input image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

    return transform(img).unsqueeze(0)  # Add batch dimension


def predict_single(model, image_path, class_names):
    """Run inference on a single image."""
    tensor = load_image(image_path)
    if tensor is None:
        return None

    with torch.no_grad():
        output = model(tensor)
        _, pred = torch.max(output, 1)
        predicted_class = class_names[pred.item()]

    logger.info(f"Image: {image_path} → Predicted class: {predicted_class}")
    return predicted_class


def predict_directory(model, dir_path, class_names):
    """Run inference for all images inside a directory and save results."""
    dir_path = Path(dir_path)

    if not dir_path.exists() or not dir_path.is_dir():
        logger.error(f"Invalid directory: {dir_path}")
        return

    results = []

    for filename in sorted(dir_path.iterdir()):
        if filename.suffix.lower() not in [".jpg", ".png", ".jpeg", ".bmp"]:
            continue

        pred = predict_single(model, filename, class_names)
        results.append({"filename": filename.name, "prediction": pred})

    df = pd.DataFrame(results)
    out_csv = OUTPUT_DIR / "inference_results.csv"
    df.to_csv(out_csv, index=False)

    logger.info(f"Inference completed. Results saved to: {out_csv}")
    return df


def main():
    logger.info("--- 4. INFERENCE STARTED ---")

    CLASS_NAMES = ["Pronacio", "Neutralis", "Szupinacio"]

    model = load_model()
    if model is None:
        return

    dir_path = DATA_DIR / "inference"
    predict_directory(model, dir_path, CLASS_NAMES)


if __name__ == "__main__":
    main()
