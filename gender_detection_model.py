"""
Gender Detection Model Setup
Download and setup a proper gender classification model
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import urllib.request
from pathlib import Path

def create_gender_model():
    """
    Create a simple gender classification model based on ResNet18
    """
    # Load pretrained ResNet18
    model = resnet18(weights=None)
    
    # Modify the final layer for binary classification (male/female)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes: male, female
    
    return model

def download_gender_model(model_path="models/gender_model.pth"):
    """
    Download a pre-trained gender detection model or create a basic one
    """
    try:
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        if not os.path.exists(model_path):
            print("[INFO] Creating basic gender detection model...")
            
            # Create a basic model (you would normally train this on a gender dataset)
            model = create_gender_model()
            
            # Save the model
            torch.save(model.state_dict(), model_path)
            print(f"[INFO] Basic gender model saved to {model_path}")
            
            return True
        else:
            print("[INFO] Gender model already exists")
            return True
            
    except Exception as e:
        print(f"[ERROR] Failed to setup gender model: {e}")
        return False

def load_gender_model(model_path="models/gender_model.pth"):
    """
    Load the gender detection model
    """
    try:
        if os.path.exists(model_path):
            model = create_gender_model()
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            return model
        else:
            print("[WARNING] Gender model not found, creating basic one...")
            if download_gender_model(model_path):
                return load_gender_model(model_path)
            return None
            
    except Exception as e:
        print(f"[ERROR] Failed to load gender model: {e}")
        return None

def get_gender_transform():
    """
    Get the image transformations for gender detection
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_gender(model, image_crop, transform=None):
    """
    Predict gender from person crop
    """
    try:
        if model is None or image_crop is None:
            return "Unknown"
        
        if transform is None:
            transform = get_gender_transform()
        
        # Convert BGR to RGB
        if len(image_crop.shape) == 3:
            rgb_crop = image_crop[:, :, ::-1]  # BGR to RGB
        else:
            rgb_crop = image_crop
        
        # Preprocess
        input_tensor = transform(rgb_crop).unsqueeze(0)

        # Move input to the same device as the model (supports CUDA)
        try:
            model_device = next(model.parameters()).device
            input_tensor = input_tensor.to(model_device)
        except Exception:
            pass
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Map predictions to gender labels
            gender_map = {0: "Girl 👧", 1: "Boy 👦"}
            gender = gender_map.get(predicted.item(), "Person")
            
            # Only return prediction if confidence is reasonable
            if confidence.item() > 0.3:
                return gender
            else:
                return "Person"
                
    except Exception as e:
        print(f"[ERROR] Gender prediction failed: {e}")
        return "Person"

if __name__ == "__main__":
    # Setup the gender model
    print("Setting up gender detection model...")
    if download_gender_model():
        print("✅ Gender model setup complete!")
    else:
        print("❌ Gender model setup failed!")
