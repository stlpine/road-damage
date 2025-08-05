import os
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO

def test_model():
    """
    Tests a trained YOLOv8 model on each country's test set.
    Reads all configuration from a .env file and outputs in the specified
    'class x y w h confidence' format.
    """
    # --- 1. Load Configuration from .env file ---
    load_dotenv()

    # NOTE: You must add these variables to your .env file before running.
    # Example:
    # MODEL_TO_TEST_PATH=runs/train/20250805_150800_yolov8n_AdamW_e50_b16_lr0.001/weights/best.pt
    # TEST_OUTPUT_DIR=runs/detect
    # CONFIDENCE_THRESHOLD=0.25

    model_to_test_path = os.getenv('MODEL_TO_TEST_PATH')
    dataset_dir = os.getenv('DATASET_DIR')
    test_output_dir = os.getenv('TEST_OUTPUT_DIR', 'runs/detect')
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

    print("--- Test Configuration Loaded ---")
    print(f"Using Device: {device.upper()}")
    print(f"Model to Test: {model_to_test_path}")
    print(f"Dataset Path: {dataset_dir}")
    print(f"Test Output Path: {test_output_dir}")
    print(f"Confidence Threshold: {confidence_threshold}")
    print("---------------------------------\n")

    model_weights = Path(model_to_test_path)
    main_dataset_path = Path(dataset_dir)
    
    # The output will be saved in a directory named after the experiment run
    experiment_name = model_weights.parent.parent.name 
    main_output_path = Path(test_output_dir) / experiment_name

    # --- 3. Load the trained model ---
    print(f"Loading model from {model_weights}...")
    model = YOLO(model_weights)

    # Find country directories
    country_dirs = [d for d in main_dataset_path.iterdir() if d.is_dir()]
    
    if not country_dirs:
        print("Error: No country directories found in the dataset folder.")
        return

    # --- 4. Iterate through each country and run tests ---
    for country_dir in country_dirs:
        country_name = country_dir.name
        test_images_path = country_dir / 'test' / 'images'
        
        if not test_images_path.exists():
            print(f"\nSkipping '{country_name}': No 'test/images' directory found.")
            continue
            
        print(f"\n--- Processing test set for: {country_name} ---")
        
        # Define a specific output directory for this country's results
        country_output_dir = main_output_path / country_name
        country_output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(test_images_path.glob('*.jpg'))
        
        if not image_files:
            print(f"No test images found in {test_images_path}")
            continue

        for image_path in image_files:
            # Run inference on the image
            results = model(image_path, device=device, verbose=False)

            # The output file will have the same name as the image but with a .txt extension
            output_filename = image_path.with_suffix('.txt').name
            output_filepath = country_output_dir / output_filename

            with open(output_filepath, 'w') as f:
                for result in results:
                    for box in result.boxes:
                        if box.conf[0] > confidence_threshold:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            xywhn = box.xywhn[0]
                            x_center, y_center, width, height = xywhn
                            
                            # CORRECTED: Swapped confidence to be the last element
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}\n")
                            
        print(f"Detection for '{country_name}' complete. Results saved to {country_output_dir}")

if __name__ == '__main__':
    # --- Run the testing process ---
    test_model()
