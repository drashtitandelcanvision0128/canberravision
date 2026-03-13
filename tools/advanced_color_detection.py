"""
Advanced Color Detection Module using YOLO26 + MobileNetV2
High accuracy object color detection with training capabilities
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import cv2
import numpy as np
import json
import os
from PIL import Image
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime

class AdvancedColorDetector:
    """
    Advanced color detection using MobileNetV2 for high accuracy
    Supports training and fine-tuning for better color recognition
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.transforms = None
        self.color_classes = [
            'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink',
            'brown', 'black', 'white', 'gray', 'silver', 'gold', 'maroon', 'navy',
            'teal', 'olive', 'lime', 'aqua', 'fuchsia', 'silver_gray'
        ]
        self.confidence_threshold = 0.7
        self._load_model()
        self._create_transforms()
        
    def _load_model(self):
        """Load MobileNetV2 model for color classification"""
        try:
            # Load pretrained MobileNetV2
            self.model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
            
            # Modify final layer for our color classes
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, len(self.color_classes))
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            # Try to load trained weights if available
            model_path = os.path.join(os.getcwd(), "models", "color_detector_mobilenetv2.pth")
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.confidence_threshold = checkpoint.get('confidence_threshold', 0.7)
                    print(f"[INFO] Loaded trained color detector from {model_path}")
                except Exception as e:
                    print(f"[WARNING] Could not load trained weights: {e}")
                    print("[INFO] Using pretrained MobileNetV2 for color detection")
            else:
                print("[INFO] Using pretrained MobileNetV2 for color detection")
                
        except Exception as e:
            print(f"[ERROR] Failed to load MobileNetV2: {e}")
            self.model = None
    
    def _create_transforms(self):
        """Create image transforms for MobileNetV2"""
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _preprocess_image(self, image_bgr: np.ndarray) -> torch.Tensor:
        """Preprocess BGR image for MobileNetV2"""
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL
            pil_image = Image.fromarray(image_rgb)
            
            # Apply transforms
            tensor = self.transforms(pil_image).unsqueeze(0).to(self.device)
            
            return tensor
        except Exception as e:
            print(f"[ERROR] Image preprocessing failed: {e}")
            return None
    
    def detect_color_with_confidence(self, image_bgr: np.ndarray) -> Dict[str, float]:
        """
        Detect object color with confidence scores using MobileNetV2
        
        Args:
            image_bgr: Input image in BGR format
            
        Returns:
            Dictionary with color names and confidence scores
        """
        if self.model is None:
            return {"unknown": 1.0}
        
        try:
            # Preprocess image
            input_tensor = self._preprocess_image(image_bgr)
            if input_tensor is None:
                return {"unknown": 1.0}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                probs = probabilities.cpu().numpy()[0]
            
            # Create color confidence dictionary
            color_confidences = {}
            for i, color in enumerate(self.color_classes):
                confidence = float(probs[i])
                color_confidences[color] = confidence
            
            # Sort by confidence
            sorted_colors = sorted(color_confidences.items(), key=lambda x: x[1], reverse=True)
            
            # Apply confidence threshold
            if sorted_colors[0][1] < self.confidence_threshold:
                return {"unknown": sorted_colors[0][1], "detected": sorted_colors[0][0]}
            
            return dict(sorted_colors[:5])  # Return top 5 colors with confidences
            
        except Exception as e:
            print(f"[ERROR] Color detection failed: {e}")
            return {"unknown": 1.0}
    
    def get_dominant_color(self, image_bgr: np.ndarray) -> str:
        """
        Get the dominant color name with highest confidence
        
        Args:
            image_bgr: Input image in BGR format
            
        Returns:
            Dominant color name
        """
        color_confidences = self.detect_color_with_confidence(image_bgr)
        
        if "unknown" in color_confidences and len(color_confidences) == 1:
            return "unknown"
        
        # Return color with highest confidence
        best_color = max(color_confidences.items(), key=lambda x: x[1])
        return best_color[0]
    
    def enhance_with_traditional_methods(self, image_bgr: np.ndarray) -> Dict[str, any]:
        """
        Enhance MobileNetV2 predictions with traditional color detection methods
        
        Args:
            image_bgr: Input image in BGR format
            
        Returns:
            Enhanced color detection results
        """
        # Get MobileNetV2 prediction
        mobilenet_result = self.detect_color_with_confidence(image_bgr)
        
        # Get traditional HSV-based detection
        traditional_color = self._detect_color_traditional(image_bgr)
        
        # Combine results
        enhanced_result = {
            "mobilenet_prediction": mobilenet_result,
            "traditional_prediction": traditional_color,
            "final_color": None,
            "confidence": 0.0,
            "method": "hybrid"
        }
        
        # If MobileNetV2 is confident, use it
        best_mobilenet = max(mobilenet_result.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
        if isinstance(best_mobilenet[1], (int, float)) and best_mobilenet[1] > self.confidence_threshold:
            enhanced_result["final_color"] = best_mobilenet[0]
            enhanced_result["confidence"] = best_mobilenet[1]
        else:
            # Fall back to traditional method
            enhanced_result["final_color"] = traditional_color
            enhanced_result["confidence"] = 0.6
            enhanced_result["method"] = "traditional_fallback"
        
        return enhanced_result
    
    def _detect_color_traditional(self, image_bgr: np.ndarray) -> str:
        """Traditional HSV-based color detection as fallback"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            
            # Define color ranges
            color_ranges = {
                'red': [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
                'orange': [(10, 50, 50), (25, 255, 255)],
                'yellow': [(25, 50, 50), (35, 255, 255)],
                'green': [(35, 50, 50), (85, 255, 255)],
                'blue': [(100, 50, 50), (130, 255, 255)],
                'purple': [(130, 50, 50), (170, 255, 255)],
                'black': [(0, 0, 0), (180, 255, 50)],
                'white': [(0, 0, 200), (180, 30, 255)],
                'gray': [(0, 0, 50), (180, 30, 200)]
            }
            
            max_pixels = 0
            detected_color = 'unknown'
            
            for color, ranges in color_ranges.items():
                pixel_count = 0
                for i in range(0, len(ranges), 2):
                    lower = np.array(ranges[i])
                    upper = np.array(ranges[i+1])
                    mask = cv2.inRange(hsv, lower, upper)
                    pixel_count += cv2.countNonZero(mask)
                
                if pixel_count > max_pixels:
                    max_pixels = pixel_count
                    detected_color = color
            
            return detected_color
            
        except Exception as e:
            print(f"[ERROR] Traditional color detection failed: {e}")
            return "unknown"
    
    def create_training_dataset(self, image_folder: str, label_file: str) -> Tuple[List, List]:
        """
        Create training dataset from images and labels
        
        Args:
            image_folder: Folder containing training images
            label_file: JSON file with color labels
            
        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []
        
        try:
            # Load labels
            with open(label_file, 'r') as f:
                color_labels = json.load(f)
            
            # Process images
            for filename in os.listdir(image_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(image_folder, filename)
                    if filename in color_labels:
                        label = color_labels[filename].lower()
                        if label in self.color_classes:
                            image_paths.append(image_path)
                            labels.append(label)
            
            print(f"[INFO] Created dataset with {len(image_paths)} training samples")
            return image_paths, labels
            
        except Exception as e:
            print(f"[ERROR] Failed to create training dataset: {e}")
            return [], []
    
    def train_model(self, image_paths: List[str], labels: List[str], 
                   epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001):
        """
        Train the MobileNetV2 color detection model
        
        Args:
            image_paths: List of training image paths
            labels: List of corresponding color labels
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
        """
        if self.model is None:
            print("[ERROR] Model not loaded for training")
            return
        
        try:
            print(f"[INFO] Starting color detection model training...")
            print(f"[INFO] Training samples: {len(image_paths)}")
            print(f"[INFO] Epochs: {epochs}, Batch size: {batch_size}")
            
            # Create dataset and dataloader
            from torch.utils.data import Dataset, DataLoader
            
            class ColorDataset(Dataset):
                def __init__(self, image_paths, labels, transforms, color_classes):
                    self.image_paths = image_paths
                    self.labels = labels
                    self.transforms = transforms
                    self.color_classes = color_classes
                    self.class_to_idx = {color: idx for idx, color in enumerate(color_classes)}
                
                def __len__(self):
                    return len(self.image_paths)
                
                def __getitem__(self, idx):
                    image_path = self.image_paths[idx]
                    label = self.labels[idx]
                    
                    # Load image
                    image = cv2.imread(image_path)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                    
                    # Apply transforms
                    if self.transforms:
                        image_tensor = self.transforms(pil_image)
                    
                    # Convert label to index
                    label_idx = self.class_to_idx[label]
                    
                    return image_tensor, label_idx
            
            # Create dataset
            dataset = ColorDataset(image_paths, labels, self.transforms, self.color_classes)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
            
            # Training loop
            self.model.train()
            best_loss = float('inf')
            
            for epoch in range(epochs):
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_idx, (images, targets) in enumerate(dataloader):
                    images, targets = images.to(self.device), targets.to(self.device)
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Statistics
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                    if batch_idx % 10 == 0:
                        print(f"[INFO] Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
                scheduler.step()
                
                avg_loss = total_loss / len(dataloader)
                accuracy = 100 * correct / total
                
                print(f"[INFO] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
                
                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self._save_model(epoch, avg_loss, accuracy)
            
            print(f"[INFO] Training completed! Best loss: {best_loss:.4f}")
            
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
    
    def _save_model(self, epoch: int, loss: float, accuracy: float):
        """Save the trained model"""
        try:
            models_dir = os.path.join(os.getcwd(), "models")
            os.makedirs(models_dir, exist_ok=True)
            
            model_path = os.path.join(models_dir, "color_detector_mobilenetv2.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'loss': loss,
                'accuracy': accuracy,
                'confidence_threshold': self.confidence_threshold,
                'color_classes': self.color_classes,
                'timestamp': datetime.now().isoformat()
            }, model_path)
            
            print(f"[INFO] Model saved to {model_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")
    
    def evaluate_accuracy(self, test_images: List[str], test_labels: List[str]) -> Dict[str, float]:
        """
        Evaluate model accuracy on test dataset
        
        Args:
            test_images: List of test image paths
            test_labels: List of true color labels
            
        Returns:
            Dictionary with accuracy metrics
        """
        if self.model is None:
            return {"accuracy": 0.0}
        
        try:
            correct = 0
            total = len(test_images)
            color_correct = {color: 0 for color in self.color_classes}
            color_total = {color: 0 for color in self.color_classes}
            
            print(f"[INFO] Evaluating model on {total} test images...")
            
            for image_path, true_label in zip(test_images, test_labels):
                # Load and predict
                image = cv2.imread(image_path)
                predicted_color = self.get_dominant_color(image)
                
                # Update statistics
                if predicted_color == true_label.lower():
                    correct += 1
                    color_correct[true_label.lower()] += 1
                
                color_total[true_label.lower()] += 1
            
            # Calculate metrics
            overall_accuracy = (correct / total) * 100
            color_accuracies = {}
            
            for color in self.color_classes:
                if color_total[color] > 0:
                    color_accuracies[color] = (color_correct[color] / color_total[color]) * 100
                else:
                    color_accuracies[color] = 0.0
            
            results = {
                "overall_accuracy": overall_accuracy,
                "color_accuracies": color_accuracies,
                "total_tested": total,
                "correct_predictions": correct
            }
            
            print(f"[INFO] Overall Accuracy: {overall_accuracy:.2f}%")
            for color, acc in color_accuracies.items():
                if color_total[color] > 0:
                    print(f"[INFO] {color.capitalize()}: {acc:.2f}% ({color_correct[color]}/{color_total[color]})")
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Evaluation failed: {e}")
            return {"accuracy": 0.0}


# Global instance for easy access
advanced_color_detector = None

def get_advanced_color_detector() -> AdvancedColorDetector:
    """Get or create the advanced color detector instance"""
    global advanced_color_detector
    if advanced_color_detector is None:
        advanced_color_detector = AdvancedColorDetector()
    return advanced_color_detector

def detect_object_color_advanced(image_bgr: np.ndarray, use_hybrid: bool = True) -> Dict[str, any]:
    """
    Detect object color using advanced MobileNetV2-based method
    
    Args:
        image_bgr: Input image in BGR format
        use_hybrid: Whether to use hybrid approach with traditional methods
        
    Returns:
        Dictionary with color detection results
    """
    detector = get_advanced_color_detector()
    
    if use_hybrid:
        return detector.enhance_with_traditional_methods(image_bgr)
    else:
        color_confidences = detector.detect_color_with_confidence(image_bgr)
        best_color = max(color_confidences.items(), key=lambda x: x[1])
        
        return {
            "final_color": best_color[0],
            "confidence": best_color[1],
            "all_confidences": color_confidences,
            "method": "mobilenetv2_only"
        }

if __name__ == "__main__":
    print("Advanced Color Detection Module")
    print("Features:")
    print("- MobileNetV2-based color classification")
    print("- Hybrid approach with traditional methods")
    print("- Training and fine-tuning capabilities")
    print("- High accuracy color detection")
    print("- Support for 22 different colors")
    print("\nUsage:")
    print("1. For basic detection: detect_object_color_advanced(image)")
    print("2. For training: detector.train_model(image_paths, labels)")
    print("3. For evaluation: detector.evaluate_accuracy(test_images, test_labels)")
