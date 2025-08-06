import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO

# --- Weighted Boxes Fusion (WBF) Implementation ---
def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0):
    if weights is None:
        weights = [1.0] * len(boxes_list)
    all_boxes, all_scores, all_labels = [], [], []
    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            if scores_list[i][j] >= skip_box_thr:
                all_boxes.append(boxes_list[i][j])
                all_scores.append(scores_list[i][j] * weights[i])
                all_labels.append(labels_list[i][j])
    if not all_boxes:
        return np.array([]), np.array([]), np.array([])
    all_boxes, all_scores, all_labels = np.array(all_boxes), np.array(all_scores), np.array(all_labels)
    indices = np.argsort(all_scores)[::-1]
    all_boxes, all_scores, all_labels = all_boxes[indices], all_scores[indices], all_labels[indices]
    unique_labels = np.unique(all_labels)
    final_boxes, final_scores, final_labels = [], [], []
    for label in unique_labels:
        class_indices = np.where(all_labels == label)[0]
        class_boxes, class_scores = all_boxes[class_indices], all_scores[class_indices]
        matched = np.zeros(len(class_boxes), dtype=bool)
        for i in range(len(class_boxes)):
            if matched[i]: continue
            cluster_boxes, cluster_scores = [class_boxes[i]], [class_scores[i]]
            matched[i] = True
            for j in range(i + 1, len(class_boxes)):
                if matched[j]: continue
                box1, box2 = class_boxes[i], class_boxes[j]
                inter_x1, inter_y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
                inter_x2, inter_y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                area1, area2 = (box1[2] - box1[0]) * (box1[3] - box1[1]), (box2[2] - box2[0]) * (box2[3] - box2[1])
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                if iou > iou_thr:
                    matched[j] = True
                    cluster_boxes.append(class_boxes[j])
                    cluster_scores.append(class_scores[j])
            cluster_boxes, cluster_scores = np.array(cluster_boxes), np.array(cluster_scores)
            total_score = np.sum(cluster_scores)
            fused_box = np.sum(cluster_boxes * cluster_scores[:, np.newaxis], axis=0) / total_score
            fused_score = total_score / len(cluster_scores)
            final_boxes.append(fused_box)
            final_scores.append(fused_score)
            final_labels.append(label)
    return np.array(final_boxes), np.array(final_scores), np.array(final_labels)

def test_ensemble():
    load_dotenv()
    team_name = os.getenv('TEAM_NAME')
    model_paths_str = os.getenv('ENSEMBLE_MODEL_PATHS')
    dataset_dir = os.getenv('DATASET_DIR')
    device = os.getenv('DEVICE', 'cpu')
    test_augment = os.getenv('TEST_AUGMENT', 'False').lower() == 'true'
    iou_threshold = float(os.getenv('IOU_THRESHOLD', 0.5))
    # Read the confidence threshold from .env, defaulting to 0.01 for ensembling
    # confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.01))
    confidence_threshold = 0.01
    img_size = int(os.getenv('IMG_SIZE', 640))
    batch_size = int(os.getenv('BATCH_SIZE', 16))

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

    print("--- Loading Ensemble Models ---")
    models = [YOLO(path) for path in model_paths]
    print(f"Loaded {len(models)} models for ensembling.")

    num_models = len(models)
    hyperparam_tag = f"img{img_size}-b{batch_size}"
    experiment_name = f"Ensemble-{num_models}models-{hyperparam_tag}"
    submission_folder = Path(f"./{team_name}/{team_name}_{experiment_name}")
    submission_folder.mkdir(parents=True, exist_ok=True)
    print(f"Creating submission folder at: {submission_folder.resolve()}")

    main_dataset_path = Path(dataset_dir)
    country_dirs = [d for d in main_dataset_path.iterdir() if d.is_dir()]
    total_predictions = 0

    for country_dir in country_dirs:
        test_images_path = country_dir / 'test' / 'images'
        if not test_images_path.exists(): continue
        print(f"   - Processing test images for: {country_dir.name}")
        image_files = list(test_images_path.glob('*.jpg'))
        
        for image_path in image_files:
            boxes_list, scores_list, labels_list = [], [], []
            for model in models:
                # CORRECTED: Use the confidence_threshold variable from the .env file
                results = model(image_path, device=device, verbose=False, augment=test_augment, iou=iou_threshold, conf=confidence_threshold)
                boxes_list.append(results[0].boxes.xyxyn.cpu().numpy())
                scores_list.append(results[0].boxes.conf.cpu().numpy())
                labels_list.append(results[0].boxes.cls.cpu().numpy())
            
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=0.55, skip_box_thr=0.1)

            output_filepath = submission_folder / image_path.with_suffix('.txt').name
            with open(output_filepath, 'w') as f:
                for i in range(len(fused_boxes)):
                    box = fused_boxes[i]
                    x_center, y_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    width, height = box[2] - box[0], box[3] - box[1]
                    f.write(f"{int(fused_labels[i])} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {fused_scores[i]:.6f}\n")
            total_predictions += 1

    print(f"\n✅ Ensemble submission generated successfully!")
    print(f"   - Total prediction files created: {total_predictions}")

if __name__ == '__main__':
    test_ensemble()
