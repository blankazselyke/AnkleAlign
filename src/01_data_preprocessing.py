import sys
import shutil
import zipfile
import pandas as pd
import logging
from pathlib import Path
import requests
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Saját modulok importálása
from config import DATA_DIR, PROCESSED_DATA_DIR, PROCESSED_IMAGES_DIR, CSV_PATH, LABEL_MAP, CSV_PATH_TRAIN, \
    CSV_PATH_TEST, CSV_PATH_VAL, VAL_SPLIT, TEST_SPLIT, TRAIN_SPLIT, SEED
from utils import setup_logger

logger = setup_logger()

DATASET_URL = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQB8kDcLEuTqQphHx7pv4Cw5AW7XMJp5MUbwortTASU223A?e=Uu6CTj&download=1&xsdata=MDV8MDJ8fDIyOTc1YmYyMWMzNzQyODFlZWZhMDhkZTM3YmNkMjdifDZhMzU0OGFiNzU3MDQyNzE5MWE4NThkYTAwNjk3MDI5fDB8MHw2MzkwMDk0ODEyNTc5MDU5MTR8VW5rbm93bnxWR1ZoYlhOVFpXTjFjbWwwZVZObGNuWnBZMlY4ZXlKRFFTSTZJbFJsWVcxelgwRlVVRk5sY25acFkyVmZVMUJQVEU5R0lpd2lWaUk2SWpBdU1DNHdNREF3SWl3aVVDSTZJbGRwYmpNeUlpd2lRVTRpT2lKUGRHaGxjaUlzSWxkVUlqb3hNWDA9fDF8TDNSbFlXMXpMekU1T2xSM1NIcHViVlpTVlVKUGFGUjFRVTFyWlc1blIyVlhTSEkzYjB0WVNWQkNTamxxTWtKbkxVdFdkMnN4UUhSb2NtVmhaQzUwWVdOMk1pOWphR0Z1Ym1Wc2N5OHhPVHBVZDBoNmJtMVdVbFZDVDJoVWRVRk5hMlZ1WjBkbFYwaHlOMjlMV0VsUVFrbzVhakpDWnkxTFZuZHJNVUIwYUhKbFlXUXVkR0ZqZGpJdmJXVnpjMkZuWlhNdk1UYzJOVE0xTVRNeU5ETTJPQT09fDBiYmVmZWIwYWJmOTRkZTFlZWZhMDhkZTM3YmNkMjdifGRlNDNhNjEyMWZmNzQxOTk4OGJiYzk4ZWMzZjU4MTdk&sdata=SWRDUWQrSVVCZTViZ05ZVEp2dU10ZFhJWG90RTdYZCtxSjBXbEtUclBCND0%3D&ovuser=6a3548ab-7570-4271-91a8-58da00697029%2Ckissblankazselyke%40edu.bme.hu"


def setup_dirs():
    if PROCESSED_DATA_DIR.exists(): shutil.rmtree(PROCESSED_DATA_DIR)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url, dest_path):
    logger.info(f"Downloading dataset from SharePoint...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

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
    zips = list(DATA_DIR.glob("*.zip"))

    if not zips:
        logger.warning(f"No zip found in {DATA_DIR}. Attempting download...")
        zip_path = DATA_DIR / "downloaded_dataset.zip"
        success = download_file(DATASET_URL, zip_path)

        if not success:
            logger.error("Could not obtain data. Exiting.")
            sys.exit(1)

        zips = [zip_path]

    logger.info(f"Unzipping {zips[0].name}...")
    try:
        with zipfile.ZipFile(zips[0], 'r') as zf:
            zf.extractall(PROCESSED_DATA_DIR)
        return PROCESSED_DATA_DIR
    except zipfile.BadZipFile:
        logger.error("The downloaded file is not a valid zip file.")
        sys.exit(1)


def clean_ls_filename(filename, neptun_code):
    candidates = [filename]
    if '-' in filename:
        candidates.append(filename.split('-', 1)[1])
        suffix = f"_{neptun_code}"
        stem = Path(filename.split('-', 1)[1]).stem
        if stem.endswith(suffix) or stem.lower().endswith(suffix.lower()):
            clean_stem = stem[:-len(suffix)]
            extension = Path(filename).suffix
            candidates.append(f"{clean_stem}{extension}")
    return candidates


def main():
    logger.info("--- 1. STEP: DATA PREPROCESSING STARTED ---")
    setup_dirs()
    working_dir = extract_source()

    # Wrapper detection
    possible_sub = working_dir / "anklealign"
    if possible_sub.exists() and possible_sub.is_dir():
        working_dir = possible_sub

    logger.info(f"Processing data from: {working_dir}")

    student_folders = [p for p in working_dir.iterdir() if p.is_dir() and not p.name.startswith("__")]
    student_folders.sort(key=lambda x: x.name)

    dataset_entries = []

    for student_folder in student_folders:
        neptun = student_folder.name
        if "consensus" in neptun.lower() or "sample" in neptun.lower():
            continue

        file_map = {f.name: f for f in student_folder.rglob("*") if f.is_file()}
        json_files = list(student_folder.rglob("*.json"))

        if not json_files:
            logger.warning(f"No JSON in {neptun}")
            continue

        try:
            import json
            with open(json_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"JSON error in {neptun}: {e}")
            continue

        for item in data:
            orig_name = item.get('file_upload')
            if not orig_name: continue

            found_path = None
            for name in clean_ls_filename(orig_name, neptun):
                if name in file_map:
                    found_path = file_map[name]
                    break

            if not found_path: continue

            choice = None
            for ann in item.get('annotations', []):
                if not ann.get('was_cancelled') and ann.get('result'):
                    try:
                        val = ann['result'][0]['value']['choices'][0]
                        if val in LABEL_MAP: choice = val
                    except:
                        continue

            if not choice: continue

            label_id = LABEL_MAP[choice]
            new_name = f"{neptun}_{found_path.name}"
            dest = PROCESSED_IMAGES_DIR / new_name
            shutil.copy2(found_path, dest)

            # Relatív út a CSV-be
            dataset_entries.append({
                'filename': f"processed_data/images/{new_name}",
                'label': label_id
            })

    if dataset_entries:
        df = pd.DataFrame(dataset_entries)
        df.to_csv(CSV_PATH, index=False)
        logger.info(f"Data processing complete. Processed {len(df)} images.")
        logger.info(f"CSV saved to: {CSV_PATH}")
    else:
        logger.error("No images processed!")

    df = pd.DataFrame(dataset_entries)
    df.to_csv(CSV_PATH, index=False)
    logger.info(f"CSV saved: {len(df)} images.")

    logger.info("Performing Stratified Train/Val/Test split...")

    # Test Split
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SPLIT, stratify=df['label'], random_state=SEED
    )

    relative_val_size = VAL_SPLIT / (1 - TEST_SPLIT)

    # Train/Val Split
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, stratify=train_val_df['label'], random_state=SEED
    )

    # Mentés külön fájlokba
    train_df.to_csv(CSV_PATH_TRAIN, index=False)
    val_df.to_csv(CSV_PATH_VAL, index=False)
    test_df.to_csv(CSV_PATH_TEST, index=False)

    logger.info(f"Splits saved:")
    logger.info(f"  Train: {len(train_df)}")
    logger.info(f"  Val:   {len(val_df)}")
    logger.info(f"  Test:  {len(test_df)}")


if __name__ == "__main__":
    main()
