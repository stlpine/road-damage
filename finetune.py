import os
import random
import yaml
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO

def finetune_model():
    """
    Performs Stage 2 of training: fine-tuning a pre-trained base model by
    training all layers with a smaller learning rate.
    """
    # --- 1. Load Configuration from .env file ---
    load_dotenv()
    try:
        # Hardware & Paths
        device = os.getenv('DEVICE', 'cpu')
        dataset_dir = os.getenv('DATASET_DIR')
        output_dir = os.getenv('OUTPUT_DIR', 'runs/train')

        # Fine-tuning specific config
        base_model_path = os.getenv('BASE_MODEL_PATH')
        finetune_epochs = int(os.getenv('FINETUNE_EPOCHS', 30))
        finetune_lr = float(os.getenv('FINETUNE_LR', 0.0005))

        # General Config
        batch_size = int(os.getenv('BATCH_SIZE', 16))
        img_size = int(os.getenv('IMG_SIZE', 640))
        patience = int(os.getenv('PATIENCE', 25))
        optimizer = os.getenv('OPTIMIZER', 'auto')
        weight_decay = float(os.getenv('WEIGHT_DECAY', 0.0005))
        momentum = float(os.getenv('MOMENTUM', 0.937))

        # Augmentation
        hsv_h = float(os.getenv('hsv_h', 0.015))
        hsv_s = float(os.getenv('hsv_s', 0.7))
        hsv_v = float(os.getenv('hsv_v', 0.4))
        degrees = float(os.getenv('DEGREES', 0.0))
        translate = float(os.getenv('TRANSLATE', 0.1))
        scale = float(os.getenv('SCALE', 0.5))
        fliplr = float(os.getenv('FLIPLR', 0.5))
        mosaic = float(os.getenv('MOSAIC', 1.0))

    except (ValueError, TypeError) as e:
        print(f"Error reading .env file: {e}")
        return

    if not dataset_dir or not base_model_path or not Path(base_model_path).exists():
        print("Error: Ensure DATASET_DIR is set and BASE_MODEL_PATH points to a valid 'best.pt' file from your base training.")
        return

    # --- 2. Prepare Dataset and Config ---
    dataset_path = Path(dataset_dir)
    finetune_output_dir = Path(output_dir) / 'finetuned_models'
    
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

    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_ft_e{finetune_epochs}_b{batch_size}_lr{finetune_lr}"
    config_dir = finetune_output_dir / 'temp_config' / run_name
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

    # --- 3. STAGE 2: Fine-Tuning ---
    print("\n" + "="*50)
    print(f"🚀 STARTING FINE-TUNING ({finetune_epochs} epochs)")
    print(f"   - Loading base model from: {base_model_path}")
    print(f"   - Unfreezing all layers with learning rate: {finetune_lr}")
    print("="*50 + "\n")

    model = YOLO(base_model_path)
    model.train(
        data=str(data_yaml_path.resolve()),
        project=str(finetune_output_dir),
        name=run_name,
        device=device,
        epochs=finetune_epochs,
        batch=batch_size,
        imgsz=img_size,
        optimizer=optimizer,
        patience=patience,
        weight_decay=weight_decay,
        momentum=momentum,
        lr0=finetune_lr,
        freeze=0, # Unfreeze all layers
        hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v,
        degrees=degrees, translate=translate, scale=scale, fliplr=fliplr, mosaic=mosaic
    )

    print(f"\n✅ Fine-tuning complete. Final model saved in {finetune_output_dir}/{run_name}")

if __name__ == '__main__':
    finetune_model()
