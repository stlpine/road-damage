import os
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO
import torch
from torchvision.ops import nms

def test_ensemble():
    """
    Tests an ensemble of trained YOLOv8 models using Non-Maximum Suppression (NMS)
    and formats the output for submission.
    """
    # --- 1. Load Configuration from .env file ---
    load_dotenv()
    team_name = os.getenv('TEAM_NAME')
    model_paths_str = os.getenv('ENSEMBLE_MODEL_PATHS')
    dataset_dir = os.getenv('DATASET_DIR')
    device = os.getenv('DEVICE', 'cpu')
    confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.25))
    iou_threshold = float(os.getenv('IOU_THRESHOLD', 0.5))
    img_size = int(os.getenv('IMG_SIZE', 640))
    batch_size = int(os.getenv('BATCH_SIZE', 16))

    # --- 2. Validate paths and settings ---
    if not model_paths_str:
        print("Error: ENSEMBLE_MODEL_PATHS is not set in .env file.")
        return
    
    model_paths = [path.strip() for path in model_paths_str.split(',')]
    if not all(Path(p).exists() for p in model_paths):
        print("Error: One or more model paths in ENSEMBLE_MODEL_PATHS are invalid.")
        return

    if not dataset_dir or not team_name or team_name == 'YourTeamName':
        print("Error: Ensure DATASET_DIR and TEAM_NAME are correctly set.")
        return

    # --- 3. Load Models ---
    print("--- Loading Ensemble Models ---")
    models = [YOLO(path) for path in model_paths]
    print(f"Loaded {len(models)} models for ensembling.")

    # --- 4. Create Submission Folder ---
    num_models = len(models)
    hyperparam_tag = f"img{img_size}-b{batch_size}"
    experiment_name = f"Ensemble-NMS-{num_models}models-{hyperparam_tag}"
    submission_folder = Path(f"./{team_name}/{team_name}_{experiment_name}")
    submission_folder.mkdir(parents=True, exist_ok=True)
    print(f"Creating submission folder at: {submission_folder.resolve()}")

    # --- 5. Run Inference and NMS Ensembling ---
    main_dataset_path = Path(dataset_dir)
    country_dirs = [d for d in main_dataset_path.iterdir() if d.is_dir()]
    total_predictions = 0

    for country_dir in country_dirs:
        test_images_path = country_dir / 'test' / 'images'
        if not test_images_path.exists(): continue

        print(f"   - Processing test images for: {country_dir.name}")
        image_files = list(test_images_path.glob('*.jpg'))

        for image_path in image_files:
            all_boxes, all_scores, all_labels = [], [], []

            # Get predictions from each model
            for model in models:
                # Use a low confidence threshold to gather all possible boxes for NMS
                results = model(image_path, device=device, verbose=False, conf=0.01)

                all_boxes.append(results[0].boxes.xyxyn.cpu())
                all_scores.append(results[0].boxes.conf.cpu())
                all_labels.append(results[0].boxes.cls.cpu())

            # Combine all predictions into single tensors
            all_boxes = torch.cat(all_boxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # Apply NMS for each class separately
            final_boxes, final_scores, final_labels = [], [], []
            for c in torch.unique(all_labels):
                class_indices = all_labels == c
                class_boxes = all_boxes[class_indices]
                class_scores = all_scores[class_indices]
                
                # Apply NMS
                keep_indices = nms(class_boxes, class_scores, iou_threshold)
                
                final_boxes.append(class_boxes[keep_indices])
                final_scores.append(class_scores[keep_indices])
                final_labels.append(torch.full_like(class_scores[keep_indices], c))

            if final_boxes:
                final_boxes = torch.cat(final_boxes, dim=0).numpy()
                final_scores = torch.cat(final_scores, dim=0).numpy()
                final_labels = torch.cat(final_labels, dim=0).numpy()

            # Write final predictions to file
            output_filepath = submission_folder / image_path.with_suffix('.txt').name
            with open(output_filepath, 'w') as f:
                for i in range(len(final_boxes)):
                    # Filter by the final confidence threshold from .env
                    if final_scores[i] >= confidence_threshold:
                        box = final_boxes[i]
                        # Convert xyxy back to xywh
                        x_center = (box[0] + box[2]) / 2
                        y_center = (box[1] + box[3]) / 2
                        width = box[2] - box[0]
                        height = box[3] - box[1]
                        
                        f.write(f"{int(final_labels[i])} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {final_scores[i]:.6f}\n")
            
            total_predictions += 1

    print(f"\n✅ NMS Ensemble submission generated successfully!")
    print(f"   - Total prediction files created: {total_predictions}")

if __name__ == '__main__':
    test_ensemble()
