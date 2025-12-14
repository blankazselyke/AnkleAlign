import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (DATA_DIR, PROCESSED_DATA_DIR, PROCESSED_IMAGES_DIR,
                    CSV_PATH, LABEL_MAP, CSV_PATH_TRAIN, CSV_PATH_TEST,
                    CSV_PATH_VAL, VAL_SPLIT, TEST_SPLIT, SEED)
from utils import setup_logger

logger = setup_logger()

# Direct link to the dataset (SharePoint)
DATASET_URL = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQB8kDcLEuTqQphHx7pv4Cw5AW7XMJp5MUbwortTASU223A?e=Uu6CTj&download=1&xsdata=MDV8MDJ8fDIyOTc1YmYyMWMzNzQyODFlZWZhMDhkZTM3YmNkMjdifDZhMzU0OGFiNzU3MDQyNzE5MWE4NThkYTAwNjk3MDI5fDB8MHw2MzkwMDk0ODEyNTc5MDU5MTR8VW5rbm93bnxWR1ZoYlhOVFpXTjFjbWwwZVZObGNuWnBZMlY4ZXlKRFFTSTZJbFJsWVcxelgwRlVVRk5sY25acFkyVmZVMUJQVEU5R0lpd2lWaUk2SWpBdU1DNHdNREF3SWl3aVVDSTZJbGRwYmpNeUlpd2lRVTRpT2lKUGRHaGxjaUlzSWxkVUlqb3hNWDA9fDF8TDNSbFlXMXpMekU1T2xSM1NIcHViVlpTVlVKUGFGUjFRVTFyWlc1blIyVlhTSEkzYjB0WVNWQkNTamxxTWtKbkxVdFdkMnN4UUhSb2NtVmhaQzUwWVdOMk1pOWphR0Z1Ym1Wc2N5OHhPVHBVZDBoNmJtMVdVbFZDVDJoVWRVRk5hMlZ1WjBkbFYwaHlOMjlMV0VsUVFrbzVhakpDWnkxTFZuZHJNVUIwYUhKbFlXUXVkR0ZqZGpJdmJXVnpjMkZuWlhNdk1UYzJOVE0xTVRNeU5ETTJPQT09fDBiYmVmZWIwYWJmOTRkZTFlZWZhMDhkZTM3YmNkMjdifGRlNDNhNjEyMWZmNzQxOTk4OGJiYzk4ZWMzZjU4MTdk&sdata=SWRDUWQrSVVCZTViZ05ZVEp2dU10ZFhJWG90RTdYZCtxSjBXbEtUclBCND0%3D&ovuser=6a3548ab-7570-4271-91a8-58da00697029%2Ckissblankazselyke%40edu.bme.hu"


def setup_dirs():
    """
    Cleans up the previous run by removing the processed data directory.
    Recreates the directory structure for images and CSVs.
    """
    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url, dest_path):
    """
    Downloads a file from a given URL to a destination path with a progress bar.

    Args:
        url (str): The direct download link.
        dest_path (Path): The full path where the file should be saved.

    Returns:
        bool: True if successful, False otherwise.
    """
    logger.info(f"Downloading dataset...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192  # 8KB chunks

        # Initialize tqdm progress bar
        with open(dest_path, 'wb') as f, tqdm(
                desc=dest_path.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(block_size):
                size = f.write(chunk)
                bar.update(size)

        logger.info("Download complete.")
        return True
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        return False


def extract_source():
    """
    Locates the raw dataset zip file. If missing, attempts to download it.
    Extracts the contents to the processed data directory.

    Returns:
        Path: The directory where the data was extracted.
    """
    # Check if a zip file already exists in the data directory
    zips = list(DATA_DIR.glob("*.zip"))

    if not zips:
        logger.warning(f"No zip found in {DATA_DIR}. Attempting download...")
        zip_path = DATA_DIR / "downloaded_dataset.zip"

        # Trigger download
        success = download_file(DATASET_URL, zip_path)
        if not success:
            logger.error("Could not obtain data. Exiting.")
            sys.exit(1)
        zips = [zip_path]

    # Extract the first found zip file
    logger.info(f"Unzipping {zips[0].name}...")
    try:
        with zipfile.ZipFile(zips[0], 'r') as zf:
            zf.extractall(PROCESSED_DATA_DIR)
        return PROCESSED_DATA_DIR
    except zipfile.BadZipFile:
        logger.error("The file is corrupted or not a valid zip file.")
        sys.exit(1)


def clean_ls_filename(filename, neptun_code):
    """
    Generates a list of possible real filenames based on the filename string found in the JSON.
    Label Studio often adds prefixes (like hash-filename) or duplicates suffixes.

    Args:
        filename (str): The filename string from the JSON annotation.
        neptun_code (str): The student identifier (folder name).

    Returns:
        list: A list of candidate filenames to check against the file system.
    """
    candidates = [filename]

    # Label Studio often prepends a hash separated by a hyphen (e.g., "8d7s6-image.jpg")
    if '-' in filename:
        # Try removing the prefix
        candidates.append(filename.split('-', 1)[1])

        # Sometimes the Neptun code is appended awkwardly at the end
        suffix = f"_{neptun_code}"
        stem = Path(filename.split('-', 1)[1]).stem

        # If the filename ends with the neptun code twice, strip it
        if stem.endswith(suffix) or stem.lower().endswith(suffix.lower()):
            clean_stem = stem[:-len(suffix)]
            extension = Path(filename).suffix
            candidates.append(f"{clean_stem}{extension}")

    return candidates


def main():
    logger.info("Step 1: Data Preprocessing Started")

    # Prepare Environment
    setup_dirs()
    working_dir = extract_source()

    # Wrapper Detection
    # Sometimes zips extract to 'root/anklealign' instead of just 'root'
    possible_sub = working_dir / "anklealign"
    if possible_sub.exists() and possible_sub.is_dir():
        working_dir = possible_sub

    logger.info(f"Processing data from: {working_dir}")

    # Iterate over Student Folders
    # Identify folders, ignoring system files like __MACOSX
    student_folders = [p for p in working_dir.iterdir() if p.is_dir() and not p.name.startswith("__")]
    student_folders.sort(key=lambda x: x.name)

    dataset_entries = []

    for student_folder in student_folders:
        neptun = student_folder.name

        # Filter out non-data folders (consensus or sample data)
        if any(x in neptun.lower() for x in ["consensus", "sample"]):
            continue

        # Map all files in the folder for quick lookup
        file_map = {f.name: f for f in student_folder.rglob("*") if f.is_file()}
        json_files = list(student_folder.rglob("*.json"))

        if not json_files:
            logger.warning(f"No JSON annotation found in {neptun}")
            continue

        try:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"JSON parsing error in {neptun}: {e}")
            continue

        # Parse Annotations
        for item in data:
            orig_name = item.get('file_upload')
            if not orig_name: continue

            # Try to find the actual image file using heuristic matching
            found_path = None
            for name in clean_ls_filename(orig_name, neptun):
                if name in file_map:
                    found_path = file_map[name]
                    break

            if not found_path: continue

            # Extract the label choice
            choice = None
            for ann in item.get('annotations', []):
                # Ensure annotation was not cancelled and has a result
                if not ann.get('was_cancelled') and ann.get('result'):
                    try:
                        val = ann['result'][0]['value']['choices'][0]
                        if val in LABEL_MAP: choice = val
                    except (IndexError, KeyError):
                        continue

            if not choice: continue

            # Standardize Data
            # Map text label to integer ID
            label_id = LABEL_MAP[choice]

            # Create a unique filename: NEPTUN_OriginalName.jpg
            new_name = f"{neptun}_{found_path.name}"
            dest = PROCESSED_IMAGES_DIR / new_name

            # Copy file to the flat processed directory
            shutil.copy2(found_path, dest)

            # Record entry for CSV
            dataset_entries.append({
                'filename': f"processed_data/images/{new_name}",
                'label': label_id
            })

    # Save Aggregated Data
    if not dataset_entries:
        logger.error("No valid images processed! Check directory structure.")
        sys.exit(1)

    df = pd.DataFrame(dataset_entries)
    df.to_csv(CSV_PATH, index=False)
    logger.info(f"Full dataset created: {len(df)} images.")

    # Stratified Split (Train / Val / Test)
    logger.info("Performing Stratified Split...")

    # First, split off the Test set (e.g., 10%)
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SPLIT, stratify=df['label'], random_state=SEED
    )

    # Calculate validation size relative to the remaining data
    relative_val_size = VAL_SPLIT / (1 - TEST_SPLIT)

    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, stratify=train_val_df['label'], random_state=SEED
    )

    # Save final splits
    train_df.to_csv(CSV_PATH_TRAIN, index=False)
    val_df.to_csv(CSV_PATH_VAL, index=False)
    test_df.to_csv(CSV_PATH_TEST, index=False)

    logger.info(f"Splits saved: Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")


if __name__ == "__main__":
    main()