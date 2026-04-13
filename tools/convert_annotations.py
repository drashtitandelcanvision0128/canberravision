"""
Annotation Conversion Tool for PPE Datasets
Converts various annotation formats to YOLO format

Supports:
- COCO JSON to YOLO
- Pascal VOC XML to YOLO
- CSV to YOLO
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import csv
from typing import Dict, List, Tuple

class AnnotationConverter:
    """Converts various annotation formats to YOLO"""
    
    def __init__(self, class_mapping: Dict[str, int]):
        """
        Args:
            class_mapping: Dict mapping class names to class IDs
                          e.g., {'helmet': 0, 'head': 1, 'seatbelt': 2}
        """
        self.class_mapping = class_mapping
    
    def coco_to_yolo(self, coco_json_path: Path, output_dir: Path, image_dir: Path):
        """Convert COCO format to YOLO"""
        print(f"[INFO] Converting COCO: {coco_json_path}")
        
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Build category mapping
        cat_mapping = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Group annotations by image
        image_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        # Create image ID to filename mapping
        image_info = {img['id']: img for img in coco_data['images']}
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for img_id, annotations in image_annotations.items():
            img_data = image_info.get(img_id)
            if not img_data:
                continue
            
            img_w = img_data['width']
            img_h = img_data['height']
            img_filename = img_data['file_name']
            
            # Create YOLO label file
            label_path = output_dir / f"{Path(img_filename).stem}.txt"
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    # Get bbox (COCO: x, y, width, height)
                    x, y, w, h = ann['bbox']
                    
                    # Convert to YOLO format (normalized center x, center y, width, height)
                    x_center = (x + w / 2) / img_w
                    y_center = (y + h / 2) / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h
                    
                    # Get class ID
                    cat_name = cat_mapping.get(ann['category_id'], 'unknown')
                    class_id = self.class_mapping.get(cat_name, 0)
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        print(f"[SUCCESS] Converted {len(image_annotations)} images to YOLO format")
    
    def voc_to_yolo(self, xml_path: Path, output_dir: Path):
        """Convert Pascal VOC XML to YOLO"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)
        
        # Get filename
        filename = root.find('filename').text
        label_path = output_dir / f"{Path(filename).stem}.txt"
        
        with open(label_path, 'w') as f:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                class_id = self.class_mapping.get(class_name, 0)
                
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Convert to YOLO
                x_center = ((xmin + xmax) / 2) / img_w
                y_center = ((ymin + ymax) / 2) / img_h
                w = (xmax - xmin) / img_w
                h = (ymax - ymin) / img_h
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    
    def batch_voc_to_yolo(self, xml_dir: Path, output_dir: Path):
        """Batch convert VOC XML files"""
        print(f"[INFO] Converting VOC annotations from: {xml_dir}")
        
        xml_files = list(xml_dir.glob("*.xml"))
        for xml_file in xml_files:
            self.voc_to_yolo(xml_file, output_dir)
        
        print(f"[SUCCESS] Converted {len(xml_files)} VOC files to YOLO")

def create_sample_helmet_dataset(output_dir: Path, num_samples: int = 100):
    """Create sample helmet dataset structure for testing"""
    print(f"[INFO] Creating sample helmet dataset...")
    
    output_dir = Path(output_dir)
    
    for split in ['train', 'val']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        n_samples = num_samples if split == 'train' else num_samples // 5
        
        for i in range(n_samples):
            # Create dummy label file
            label_path = output_dir / 'labels' / split / f"sample_{i:04d}.txt"
            with open(label_path, 'w') as f:
                f.write("0 0.5 0.5 0.3 0.3\n")  # helmet class
    
    # Create dataset.yaml
    yaml_content = """
path: {}
train: images/train
val: images/val
nc: 2
names:
  0: helmet
  1: no_helmet
""".format(output_dir.absolute())
    
    with open(output_dir / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"[SUCCESS] Sample dataset created at: {output_dir}")
    print("[NOTE] Replace dummy images with real images before training!")

def main():
    """Example usage"""
    # Example: Convert helmet dataset annotations
    helmet_mapping = {
        'helmet': 0,
        'head': 1,  # no helmet
        'with_helmet': 0,
        'without_helmet': 1
    }
    
    converter = AnnotationConverter(helmet_mapping)
    
    # Example paths (adjust to your dataset)
    # converter.coco_to_yolo(
    #     Path("ppe_dataset/raw/helmet/annotations.json"),
    #     Path("ppe_dataset/helmet/organized/labels/all"),
    #     Path("ppe_dataset/raw/helmet/images")
    # )
    
    # Create sample dataset for testing
    # create_sample_helmet_dataset(Path("ppe_dataset/sample"), num_samples=50)
    
    print("[INFO] Annotation converter ready!")
    print("Uncomment the conversion code above and adjust paths for your dataset.")

if __name__ == "__main__":
    main()
