import os
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO

def test_model():
    """
    Tests a trained YOLOv8 model on each country's test set and formats
    the output for official submission using the TEAM_NAME.
    """
    # --- 1. Load Configuration from .env file ---
    load_dotenv()

    # NOTE: You must add these variables to your .env file before running.
    # Example:
    # TEAM_NAME=MyAwesomeTeam
    # MODEL_TO_TEST_PATH=runs/train/finetuned_models/your_run/weights/best.pt
    # CONFIDENCE_THRESHOLD=0.25

    team_name = os.getenv('TEAM_NAME')
    model_to_test_path = os.getenv('MODEL_TO_TEST_PATH')
    dataset_dir = os.getenv('DATASET_DIR')
    confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.25))
    device = os.getenv('DEVICE', 'cpu')

    # --- 2. Validate paths and settings ---
    if not model_to_test_path or not Path(model_to_test_path).exists():
        print(f"Error: MODEL_TO_TEST_PATH is not set in .env or the file does not exist.")
        print("Please set it to the 'best.pt' file from a training run.")
        return

    if not dataset_dir:
        print("Error: DATASET_DIR is not set in the .env file.")
        return

    if not team_name or team_name == 'YourTeamName':
        print("Error: TEAM_NAME is not correctly set in your .env file.")
        return

    print("--- Test & Submission Generation ---")
    print(f"Team Name: {team_name}")
    print(f"Using Device: {device.upper()}")
    print(f"Model to Test: {model_to_test_path}")
    print(f"Dataset Path: {dataset_dir}")
    print(f"Confidence Threshold: {confidence_threshold}")
    print("------------------------------------\n")

    model_weights = Path(model_to_test_path)
    main_dataset_path = Path(dataset_dir)

    # --- 3. Create the final submission folder structure ---
    # Get the experiment name from the parent directory of the model weights
    experiment_name = model_weights.parent.parent.name
    # Clean the experiment name to remove invalid characters for folder names
    experiment_name_clean = experiment_name.replace('_', '-')

    submission_folder = Path(f"./{team_name}/{team_name}_{experiment_name_clean}")
    submission_folder.mkdir(parents=True, exist_ok=True)

    print(f"Creating submission folder at: {submission_folder.resolve()}")

    # --- 4. Load the trained model ---
    print(f"Loading model from {model_weights}...")
    model = YOLO(model_weights)

    # Find country directories
    country_dirs = [d for d in main_dataset_path.iterdir() if d.is_dir()]

    if not country_dirs:
        print("Error: No country directories found in the dataset folder.")
        return

    # --- 5. Iterate through each country and generate prediction files ---
    total_predictions = 0
    for country_dir in country_dirs:
        test_images_path = country_dir / 'test' / 'images'
        if not test_images_path.exists():
            continue

        print(f"   - Processing test images for: {country_dir.name}")
        image_files = list(test_images_path.glob('*.jpg'))

        for image_path in image_files:
            # Run inference on the image
            results = model(image_path, device=device, verbose=False)

            # The output file will have the same name as the image but with a .txt extension
            output_filename = image_path.with_suffix('.txt').name
            output_filepath = submission_folder / output_filename

            with open(output_filepath, 'w') as f:
                for result in results:
                    for box in result.boxes:
                        if box.conf[0] > confidence_threshold:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            xywhn = box.xywhn[0]
                            x_center, y_center, width, height = xywhn
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}\n")
            total_predictions += 1

    print(f"\n✅ Submission folder generated successfully!")
    print(f"   - Total prediction files created: {total_predictions}")
    print(f"   - Your submission is ready in the '{team_name}' folder.")

if __name__ == '__main__':
    # --- Run the testing process ---
    test_model()
