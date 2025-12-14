# AnkleAlign project

## Project Details

### Project Information

- **Selected Topic**: AnkleAlign
- **Student Name**: Blanka Zselyke Kiss
- **Aiming for +1 Mark**: No

### Solution Description

#### **Problem Statement**

The goal of this project is to automate the classification of ankle alignment from foot images into three clinical categories: **Pronation**, **Neutral**, and **Supination**. The primary challenge was the limited dataset size sourced from student uploads, which contained significant variance in lighting, framing, and background, as well as class imbalance.

#### **Model Architecture**

1.  **Baseline Model:** A lightweight, custom CNN trained from scratch (3 convolutional blocks with MaxPool and 2 fully connected layers) to establish a performance benchmark.
2.  **Main Model:** A Transfer Learning approach utilizing a **ResNet18** backbone pre-trained on ImageNet. I replaced the final classification head with a linear layer outputting 3 classes. I chose this architecture for its ability to extract robust features despite the small dataset size.

#### **Training Methodology**

* **Data Preprocessing:** An automated pipeline extracts images from raw Label Studio exports.
    * **Cleaning:** I explicitly excluded folders containing "consensus" or "sample" data to ensure dataset integrity. Additionally, two folders (`ECSGGY` and `GI9Y8B`) were skipped as they contained no JSON annotations. After filtering, the usable dataset consisted of **323 images**.
* **Data Splitting:** I employed a Stratified **80-10-10 split** to ensure class balance across all subsets. This resulted in the following distribution:
    * **Train:** 257 samples
    * **Validation:** 33 samples
    * **Test:** 33 samples
* **Standardization:** All images were resized to **224x224** pixels to ensure consistency and meet the input requirements of the CNN architectures.
* **Data Augmentation:** I implemented a custom `AugmentedAnkleDataset` wrapper that expands the training set by a factor of 3. Each image is fed into the model with three distinct views: (1) a clean center crop, (2) rotation/color jitter, and (3) affine/blur transformations. This increased the effective training samples from **257** to **771**.
* **Optimization:** I trained the model using the **Adam** optimizer with Cross-Entropy Loss and **Weight Decay** (L2 regularization) to further prevent overfitting. I utilized a **ReduceLROnPlateau** scheduler to dynamically adjust the learning rate based on validation loss, coupled with **Early Stopping** to save the best-performing model checkpoint.

#### **Results**

The table below compares the performance of the custom CNN trained from scratch (Baseline) against the ResNet18 model pre-trained on ImageNet (Main Model) on the test set.

| Metric | Baseline Model (Custom CNN) | Main Model (ResNet18) | Improvement |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | 45.45% | 72.73% | +27.28% |
| **Supination F1-Score** | 0.00 | 0.60 | +0.60 |
| **Weighted Avg Precision** | 0.40 | 0.73 | +0.33 |
| **Weighted Avg Recall** | 0.45 | 0.73 | +0.28 |
| **Trainable Parameters** | ~3.2M | ~11.2M | - |

#### **Detailed Analysis**

The **Baseline Model**, consisting of a simple custom architecture with three convolutional blocks, exhibited significant issues with generalization. While it achieved high accuracy on the training set (indicating the capacity to memorize), its validation and test performance stagnated around 45%. The most critical failure of this model was its complete inability to identify the minority class, **Supination**. The model achieved a Recall of 0.00 for this category, failing to correctly classify a single Supination case in the test set. Effectively, the baseline reduced the problem to a binary classification task (Pronation vs. Neutral) with limited success, proving that training from scratch on just 257 images was insufficient for learning the distinct visual features of the minority class.

In contrast, the **Main Model (ResNet18)**, utilizing weights pre-trained on ImageNet, achieved a drastic improvement with a **72.73%** test accuracy. The most significant breakthrough was the **successful recovery of the Supination class**: the model achieved an F1-score of 0.60 for this category, proving it could extract robust features even from limited examples. Analysis of the Confusion Matrix reveals that the majority of errors occurred between visually similar, adjacent categories (e.g., distinguishing *Neutral* from *Pronation*), which is often subjective even in clinical settings.


### Data Preparation

Data preparation is executed by the src/01-data-preprocessing.py script. It checks the local data/ directory and automatically downloads the dataset from the AnkleAlign SharePoint folder if the data folder is empty. The script excludes "sample" and "consensus" folders, and skips entries with missing JSON annotations or images. Valid files are renamed using the {NEPTUN}_{Filename} pattern to prevent naming collisions.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the solution, use the following command. You must mount your local **data** directory to `/app/data` and your **output** directory to `/app/output` inside the container.

```bash
docker run -v /absolute/path/to/your/local/data:/app/data -v /absolute/path/to/your/local/output:/app/output dl-project > log/run.log 2>&1
```

*   Replace `/absolute/path/to/your/local/data` with the absolute path to the folder on your host machine where you want the dataset to be downloaded.
*   Replace `/absolute/path/to/your/local/output` with the absolute path where you want the trained model and processed files to be saved.
*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).


### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Scripts for loading, cleaning, and preprocessing the raw data.
    - `02-baseline.py`: The main script for defining the baseline model and executing the baseline training loop.
    - `02-training.py`: The main script for defining the main model and executing the main training loop.
    - `03-evaluation.py`: Scripts for evaluating the trained models on test data and generating metrics.
    - `04-inference.py`: Script for running the model on new, unseen images (located in /data/inference folder) to generate predictions.
    - `config.py`: Configuration file containing hyperparameters and paths.
    - `utils.py`: Helper functions and utilities used across different scripts.

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01-eda.ipynb`: Notebook for initial exploratory data analysis (EDA) and visualization. Also contains the distribution and properties of the target labels.

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **`data/`**:
    - `inference/`: Contains sample images used to test the inference script (`04-inference.py`) and verify model predictions on unseen data.

- **`output/`**:
    - Stores the artifacts generated during execution, including the trained model weights (`best_model.pth`, `baseline_model.pth`) and the processed CSV datasets.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
