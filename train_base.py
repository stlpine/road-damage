import os
import random
import yaml
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO

def train_base_model():
    """
    Performs Stage 1 of training: creating a base model by training on the
    custom dataset with frozen backbone layers.
    """
    # --- 1. Load Configuration from .env file ---
    load_dotenv()
    try:
        device = os.getenv('DEVICE', 'cpu')
        dataset_dir = os.getenv('DATASET_DIR')
        output_dir = os.getenv('OUTPUT_DIR', 'runs/train')
        initial_model_path = os.getenv('INITIAL_MODEL_PATH', 'yolov8n.pt')
        base_epochs = int(os.getenv('BASE_TRAINING_EPOCHS', 25))
        freeze_layers = int(os.getenv('FREEZE_LAYERS', 10))
        base_lr = float(os.getenv('BASE_LR', 0.01))
        batch_size = int(os.getenv('BATCH_SIZE', 16))
        img_size = int(os.getenv('IMG_SIZE', 640))
        optimizer = os.getenv('OPTIMIZER', 'auto')
    except (ValueError, TypeError) as e:
        print(f"Error reading .env file: {e}")
        return

    if not dataset_dir:
        print("Error: DATASET_DIR not set in .env file.")
        return

    # --- 2. Prepare Dataset and Config ---
    dataset_path = Path(dataset_dir)
    # Save base models to a specific sub-directory for clarity
    base_output_dir = Path(output_dir) / 'base_models' 
    
    all_train_images = []
    for country_dir in [d for d in dataset_path.iterdir() if d.is_dir()]:
        train_images_path = country_dir / 'train' / 'images'
        if train_images_path.exists():
            all_train_images.extend(list(train_images_path.glob('*.jpg')))

    if not all_train_images:
        print("Error: No training images found.")
        return
 
    random.shuffle(all_train_images)
    split_index = int(len(all_train_images) * 0.8)
    train_files, val_files = all_train_images[:split_index], all_train_images[split_index:]

    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(initial_model_path).stem}_base"
    config_dir = base_output_dir / 'temp_config' / run_name
    config_dir.mkdir(parents=True, exist_ok=True)

    with open(config_dir / 'train.txt', 'w') as f:
        for path in train_files: f.write(f"{path.resolve()}\n")
    with open(config_dir / 'val.txt', 'w') as f:
        for path in val_files: f.write(f"{path.resolve()}\n")

    class_names = ['Pothole', 'Alligator Crack', 'Transverse Crack', 'Longitudinal Crack']
    data_yaml_path = config_dir / 'data.yaml'
    with open(data_yaml_path, 'w') as f:
        yaml.dump({
            'train': str((config_dir / 'train.txt').resolve()),
            'val': str((config_dir / 'val.txt').resolve()),
            'nc': len(class_names),
            'names': class_names
        }, f, default_flow_style=False)

    # --- 3. STAGE 1: Base Training ---
    print("\n" + "="*50)
    print(f"🚀 STARTING BASE MODEL TRAINING ({base_epochs} epochs)")
    print(f"   - Freezing the first {freeze_layers} layers.")
    print("="*50 + "\n")

    model = YOLO(initial_model_path)
    model.train(
        data=str(data_yaml_path.resolve()),
        project=str(base_output_dir),
        name=run_name,
        device=device,
        epochs=base_epochs,
        batch=batch_size,
        imgsz=img_size,
        optimizer=optimizer,
        lr0=base_lr,
        freeze=freeze_layers
    )

    print(f"\n✅ Base model training complete. Model saved in {base_output_dir}/{run_name}")
    print(f"➡️ Next step: Update BASE_MODEL_PATH in your .env file to:")
    print(f"   {base_output_dir / run_name / 'weights' / 'best.pt'}")

if __name__ == '__main__':
    train_base_model()
