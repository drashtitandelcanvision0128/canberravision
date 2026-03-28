"""
Color Detection Training Script
Train MobileNetV2 for accurate color detection with YOLO26 integration
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

def create_sample_color_dataset():
    """
    Create sample color dataset for training
    Generates color patches and labels for training
    """
    print("[INFO] Creating sample color dataset...")
    
    # Create directories
    base_dir = "color_training_data"
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Define colors with RGB values
    colors = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'orange': (255, 165, 0),
        'purple': (128, 0, 128),
        'pink': (255, 192, 203),
        'brown': (165, 42, 42),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'gray': (128, 128, 128),
        'cyan': (0, 255, 255),
        'navy': (0, 0, 128),
        'maroon': (128, 0, 0),
        'olive': (128, 128, 0),
        'lime': (0, 255, 0),
        'aqua': (0, 255, 255),
        'fuchsia': (255, 0, 255),
        'silver': (192, 192, 192),
        'gold': (255, 215, 0),
        'teal': (0, 128, 128)
    }
    
    labels = {}
    
    # Generate color variations
    for color_name, rgb in colors.items():
        for i in range(20):  # 20 variations per color
            # Add some noise and variation
            noise_factor = 0.3
            variation = np.random.uniform(-noise_factor, noise_factor, 3)
            varied_rgb = np.clip(rgb + variation * 255, 0, 255).astype(int)
            
            # Create image with different patterns
            img_size = 224
            
            # Random pattern type
            pattern_type = np.random.choice(['solid', 'gradient', 'noise', 'texture'])
            
            if pattern_type == 'solid':
                # Solid color
                img = np.full((img_size, img_size, 3), varied_rgb, dtype=np.uint8)
                
            elif pattern_type == 'gradient':
                # Gradient effect
                img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                for y in range(img_size):
                    factor = y / img_size
                    pixel = np.clip(varied_rgb + factor * 50 - 25, 0, 255).astype(int)
                    img[y, :] = pixel
                    
            elif pattern_type == 'noise':
                # Add noise
                img = np.full((img_size, img_size, 3), varied_rgb, dtype=np.uint8)
                noise = np.random.normal(0, 20, (img_size, img_size, 3))
                img = np.clip(img + noise, 0, 255).astype(np.uint8)
                
            else:  # texture
                # Add texture pattern
                img = np.full((img_size, img_size, 3), varied_rgb, dtype=np.uint8)
                # Add some texture
                for _ in range(50):
                    x, y = np.random.randint(0, img_size, 2)
                    radius = np.random.randint(2, 10)
                    darkness = np.random.uniform(0.7, 1.3)
                    color = tuple(np.clip(varied_rgb * darkness, 0, 255).astype(int).tolist())
                    cv2.circle(img, (x, y), radius, color, -1)
            
            # Add some random brightness/contrast changes
            brightness = np.random.uniform(-30, 30)
            contrast = np.random.uniform(0.8, 1.2)
            img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
            
            # Save image
            filename = f"{color_name}_{i:03d}.jpg"
            filepath = os.path.join(images_dir, filename)
            cv2.imwrite(filepath, img)
            labels[filename] = color_name
    
    # Save labels
    labels_file = os.path.join(base_dir, "labels.json")
    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"[INFO] Created {len(labels)} training images in {images_dir}")
    print(f"[INFO] Labels saved to {labels_file}")
    
    return base_dir, labels_file

def extract_colors_from_yolo_detections(yolo_results_dir: str, output_dir: str):
    """
    Extract object colors from YOLO26 detection results for training
    
    Args:
        yolo_results_dir: Directory containing YOLO26 detection results
        output_dir: Output directory for extracted color patches
    """
    print(f"[INFO] Extracting colors from YOLO results in {yolo_results_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    labels = {}
    processed_count = 0
    
    # Process detection results
    for filename in os.listdir(yolo_results_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(yolo_results_dir, filename)
            
            try:
                with open(json_path, 'r') as f:
                    detection_data = json.load(f)
                
                # Extract image path
                image_filename = filename.replace('.json', '.jpg')
                image_path = os.path.join(yolo_results_dir, image_filename)
                
                if not os.path.exists(image_path):
                    # Try other extensions
                    for ext in ['.png', '.jpeg']:
                        image_path = os.path.join(yolo_results_dir, 
                                                filename.replace('.json', ext))
                        if os.path.exists(image_path):
                            break
                    else:
                        continue
                
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Process detected objects
                objects = detection_data.get('objects', [])
                
                for i, obj in enumerate(objects):
                    if 'bounding_box' in obj and 'color' in obj:
                        bbox = obj['bounding_box']
                        color_label = obj['color']
                        
                        if color_label.lower() in ['unknown', 'none']:
                            continue
                        
                        # Extract object patch
                        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                        
                        # Add margin
                        margin = 10
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(image.shape[1], x2 + margin)
                        y2 = min(image.shape[0], y2 + margin)
                        
                        if x2 > x1 and y2 > y1:
                            patch = image[y1:y2, x1:x2]
                            
                            # Resize to standard size
                            patch = cv2.resize(patch, (224, 224))
                            
                            # Save patch
                            patch_filename = f"{os.path.splitext(filename)[0]}_obj_{i:03d}.jpg"
                            patch_path = os.path.join(images_dir, patch_filename)
                            cv2.imwrite(patch_path, patch)
                            
                            labels[patch_filename] = color_label.lower()
                            processed_count += 1
                
            except Exception as e:
                print(f"[WARNING] Failed to process {filename}: {e}")
                continue
    
    # Save labels
    labels_file = os.path.join(output_dir, "labels.json")
    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"[INFO] Extracted {processed_count} color patches to {images_dir}")
    print(f"[INFO] Labels saved to {labels_file}")
    
    return output_dir, labels_file

def train_color_model(dataset_path: str, epochs: int = 50, batch_size: int = 32):
    """
    Train the color detection model
    
    Args:
        dataset_path: Path to training dataset
        epochs: Number of training epochs
        batch_size: Training batch size
    """
    print(f"[INFO] Training color model with dataset: {dataset_path}")
    
    try:
        from advanced_color_detection import get_advanced_color_detector
        
        # Get detector
        detector = get_advanced_color_detector()
        
        # Create dataset
        labels_file = os.path.join(dataset_path, "labels.json")
        images_dir = os.path.join(dataset_path, "images")
        
        if not os.path.exists(labels_file):
            print(f"[ERROR] Labels file not found: {labels_file}")
            return
        
        # Load training data
        image_paths, labels = detector.create_training_dataset(images_dir, labels_file)
        
        if not image_paths:
            print("[ERROR] No training data found")
            return
        
        print(f"[INFO] Starting training with {len(image_paths)} samples")
        
        # Train model
        detector.train_model(
            image_paths=image_paths,
            labels=labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=0.001
        )
        
        print("[INFO] Training completed successfully!")
        
        # Test the trained model
        print("[INFO] Testing trained model...")
        test_image = cv2.imread(image_paths[0])
        if test_image is not None:
            result = detector.enhance_with_traditional_methods(test_image)
            print(f"[INFO] Sample prediction: {result}")
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")

def evaluate_color_model(test_dataset_path: str):
    """
    Evaluate the trained color detection model
    
    Args:
        test_dataset_path: Path to test dataset
    """
    print(f"[INFO] Evaluating color model with dataset: {test_dataset_path}")
    
    try:
        from advanced_color_detection import get_advanced_color_detector
        
        # Get detector
        detector = get_advanced_color_detector()
        
        # Load test data
        labels_file = os.path.join(test_dataset_path, "labels.json")
        images_dir = os.path.join(test_dataset_path, "images")
        
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        
        test_images = []
        test_labels = []
        
        for filename, color in labels.items():
            image_path = os.path.join(images_dir, filename)
            if os.path.exists(image_path):
                test_images.append(image_path)
                test_labels.append(color)
        
        if not test_images:
            print("[ERROR] No test data found")
            return
        
        # Evaluate
        results = detector.evaluate_accuracy(test_images, test_labels)
        
        print("[INFO] Evaluation Results:")
        print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
        print(f"Total Tested: {results['total_tested']}")
        print(f"Correct Predictions: {results['correct_predictions']}")
        
        print("\nPer-color accuracies:")
        for color, accuracy in results['color_accuracies'].items():
            if accuracy > 0:
                print(f"  {color}: {accuracy:.2f}%")
        
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Color Detection Training')
    parser.add_argument('--mode', choices=['create_sample', 'extract', 'train', 'evaluate'], 
                       required=True, help='Operation mode')
    parser.add_argument('--input', help='Input path (for extract/train/evaluate)')
    parser.add_argument('--output', help='Output path (for extract/train)')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    if args.mode == 'create_sample':
        dataset_path, labels_file = create_sample_color_dataset()
        print(f"\n[INFO] Sample dataset created at: {dataset_path}")
        print(f"[INFO] To train the model, run:")
        print(f"python color_training.py --mode train --input {dataset_path} --epochs {args.epochs}")
        
    elif args.mode == 'extract':
        if not args.input or not args.output:
            print("[ERROR] --input and --output required for extract mode")
            return
        
        output_path, labels_file = extract_colors_from_yolo_detections(args.input, args.output)
        print(f"\n[INFO] Colors extracted to: {output_path}")
        print(f"[INFO] To train the model, run:")
        print(f"python color_training.py --mode train --input {output_path} --epochs {args.epochs}")
        
    elif args.mode == 'train':
        if not args.input:
            print("[ERROR] --input required for train mode")
            return
        
        train_color_model(args.input, args.epochs, args.batch_size)
        
    elif args.mode == 'evaluate':
        if not args.input:
            print("[ERROR] --input required for evaluate mode")
            return
        
        evaluate_color_model(args.input)

if __name__ == "__main__":
    main()
