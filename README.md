# Road Damage Detection and Localization

This project implements an advanced AI model to identify and localize four types of road damage: potholes, alligator cracks, longitudinal cracks, and transverse cracks. The model is built using the YOLOv8 architecture and is trained on a diverse dataset of road images from three different countries.

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Installation and Setup](#installation-and-setup)
3.  [Dataset](#dataset)
4.  [Training the Model (Two-Step Process)](#training-the-model-two-step-process)
5.  [Evaluating the Model](#evaluating-the-model)
6.  [Model Architecture and Parameters](#model-architecture-and-parameters)
7.  [Results and Interpretation](#results-and-interpretation)

---

### Project Overview

The goal of this project is to build a robust object detection model capable of accurately detecting and classifying multiple types of road damage in a single image. This is a multi-class detection and localization task.

- **Input:** Road scene images (`.jpg`).
- **Output:** For each detected damage, a text file is generated containing its class, bounding box coordinates (`x y w h`), and confidence score.

### Installation and Setup

1.  **Clone the Repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    The core dependencies are listed in `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

    **Note on PyTorch:** The `ultralytics` package will install a version of PyTorch. Ensure it is the correct version for your hardware (CPU, NVIDIA GPU, or Apple Silicon). If you encounter issues, it's best to install PyTorch manually by following the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

4.  **Configure Environment:**
    Configure your desired training parameters using a `.env` file in the root directory. An example is provided in the repository.

### Dataset

The dataset consists of thousands of annotated images from three countries, with four types of damage labeled:

- **Countries:** `country1`, `country2`, `country3`
- **Damage Types:**
  - `0`: Pothole
  - `1`: Alligator Crack
  - `2`: Transverse Crack
  - `3`: Longitudinal Crack
- **Structure:** The dataset is expected to be in a directory named `dataset` with the following structure:
  ```
  dataset/
  ├── country1/
  │   ├── train/
  │   │   ├── images/
  │   │   └── labels/
  │   └── test/
  │       └── images/
  ├── country2/
  │   └── ...
  └── country3/
      └── ...
  ```

### Training the Model (Two-Step Process)

The training process is divided into two distinct stages for optimal performance and efficient experimentation. All configuration is managed via the `.env` file.

#### Stage 1: Create the Base Model

This step is run once to create a strong base model that has learned the general features of road damage.

1.  **Configure ` .env` for Base Training:**

    - Set `INITIAL_MODEL_PATH` to a pre-trained model (e.g., `yolov8n.pt`).
    - Configure `BASE_TRAINING_EPOCHS` and `FREEZE_LAYERS`.

2.  **Run the Base Training Script:**
    ```bash
    python train_base.py
    ```
3.  **Update `.env`:** After the script finishes, it will print the path to the newly created base model (the `best.pt` file). Copy this path and paste it into the `BASE_MODEL_PATH` variable in your `.env` file.

#### Stage 2: Fine-Tune the Model

This step can be run multiple times to experiment with different fine-tuning hyperparameters.

1.  **Configure ` .env` for Fine-Tuning:**

    - Ensure `BASE_MODEL_PATH` points to the model you created in Stage 1.
    - Adjust `FINETUNE_EPOCHS`, `FINETUNE_LR`, and any other parameters you wish to experiment with.

2.  **Run the Fine-Tuning Script:**
    ```bash
    python finetune.py
    ```
    A new experiment folder will be created inside `outputs/train/finetuned_models/` containing your final model and results.

### Evaluating the Model

After training, use the `test.py` script to evaluate your final model's performance on the test sets.

1.  **Update `.env`:** Set the `MODEL_TO_TEST_PATH` variable to the path of your best fine-tuned model (e.g., `outputs/train/finetuned_models/your_finetune_run/weights/best.pt`).
2.  **Run Evaluation:**
    ```bash
    python test.py
    ```
    The script will generate a folder for each country within your `TEST_OUTPUT_DIR`, containing a `.txt` file for each test image with the predicted bounding boxes.

### Model Architecture and Parameters

- **Architecture:** This project uses **YOLOv8**, a state-of-the-art, single-stage object detection model known for its excellent balance of speed and accuracy.
- **Transfer Learning:** We employ a two-stage transfer learning approach. We start with weights pre-trained on the COCO dataset, create a specialized base model by training on our data with a frozen backbone, and then fine-tune the entire network to achieve optimal performance.
- **Hyperparameters:** All key parameters are configurable in the `.env` file. This includes learning rates for both stages, optimizer choice, batch size, image size, and extensive data augmentation settings to improve model robustness.

### Results and Interpretation

The output of the `test.py` script is a set of text files in the standard YOLO format: `class_id center_x center_y width height confidence`.

- **`class_id`**: An integer from 0-3 corresponding to the damage types.
- **`center_x, center_y, width, height`**: Bounding box coordinates, normalized to be between 0 and 1.
- **`confidence`**: The model's confidence in its prediction (0 to 1).

These raw outputs can be used to calculate standard object detection metrics like mean Average Precision (mAP), Precision, and Recall to formally evaluate the model's performance.
