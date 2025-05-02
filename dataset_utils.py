#!/usr/bin/env python3
# dataset_utils.py - Dataset management for YOLO11 training

import os
import yaml
import urllib.request
import zipfile
import shutil
from pathlib import Path

def download_dataset(url, dataset_dir='datasets/roboflow_dataset'):
    """Download YOLO formatted dataset from Roboflow"""
    print(f'Downloading dataset: {url}')

    # Create target directory
    os.makedirs(dataset_dir, exist_ok=True)

    # Download dataset
    download_url = f"{url}&format=yolov5"  # Use YOLOv5 format - more compatible directory structure
    zip_path = os.path.join(dataset_dir, 'dataset.zip')

    try:
        urllib.request.urlretrieve(download_url, zip_path)
        print(f'Download completed: {zip_path}')

        # Extract ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print(f'Archive extracted: {dataset_dir}')

        # Remove ZIP file
        os.remove(zip_path)

        # Fix directory structure
        fix_directory_structure(dataset_dir)

        # Update and save dataset YAML
        update_dataset_yaml(dataset_dir)

        return True
    except Exception as e:
        print(f'Dataset download error: {e}')
        return False

def fix_directory_structure(dataset_dir):
    """Fix directory structure to match YOLO11 expectations"""
    print(f"Checking and fixing directory structure...")

    # Roboflow structure: dataset/train/images and dataset/valid/images
    # YOLO11 expected structure: dataset/images/train and dataset/images/val

    # 1. First check existing structure and report
    train_images_dir = os.path.join(dataset_dir, 'train', 'images')
    valid_images_dir = os.path.join(dataset_dir, 'valid', 'images')
    test_images_dir = os.path.join(dataset_dir, 'test', 'images')

    train_labels_dir = os.path.join(dataset_dir, 'train', 'labels')
    valid_labels_dir = os.path.join(dataset_dir, 'valid', 'labels')
    test_labels_dir = os.path.join(dataset_dir, 'test', 'labels')

    # Expected YOLO11 structure
    yolo_images_dir = os.path.join(dataset_dir, 'images')
    yolo_labels_dir = os.path.join(dataset_dir, 'labels')

    yolo_train_images = os.path.join(yolo_images_dir, 'train')
    yolo_val_images = os.path.join(yolo_images_dir, 'val')
    yolo_test_images = os.path.join(yolo_images_dir, 'test')

    yolo_train_labels = os.path.join(yolo_labels_dir, 'train')
    yolo_val_labels = os.path.join(yolo_labels_dir, 'val')
    yolo_test_labels = os.path.join(yolo_labels_dir, 'test')

    # Check directory existence
    has_train = os.path.exists(train_images_dir) and os.path.exists(train_labels_dir)
    has_valid = os.path.exists(valid_images_dir) and os.path.exists(valid_labels_dir)
    has_test = os.path.exists(test_images_dir) and os.path.exists(test_labels_dir)

    print(f"Current structure:")
    print(f"  Train folder exists: {has_train}")
    print(f"  Valid folder exists: {has_valid}")
    print(f"  Test folder exists: {has_test}")

    # Create YOLO11 structure directories
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)

    # 2. Copy train folder
    if has_train:
        print(f"Organizing train data...")
        # If folder doesn't exist or source has files, copy them
        if not os.path.exists(yolo_train_images) or len(os.listdir(train_images_dir)) > 0:
            # Copy images
            os.makedirs(yolo_train_images, exist_ok=True)
            for img_file in os.listdir(train_images_dir):
                src_path = os.path.join(train_images_dir, img_file)
                dst_path = os.path.join(yolo_train_images, img_file)
                shutil.copy2(src_path, dst_path)

            # Copy labels
            os.makedirs(yolo_train_labels, exist_ok=True)
            for label_file in os.listdir(train_labels_dir):
                src_path = os.path.join(train_labels_dir, label_file)
                dst_path = os.path.join(yolo_train_labels, label_file)
                shutil.copy2(src_path, dst_path)

            print(f"  Train data copied: {len(os.listdir(yolo_train_images))} images, {len(os.listdir(yolo_train_labels))} labels")

    # 3. Copy valid folder (as val)
    if has_valid:
        print(f"Organizing validation data...")
        # If folder doesn't exist or source has files, copy them
        if not os.path.exists(yolo_val_images) or len(os.listdir(valid_images_dir)) > 0:
            # Copy images
            os.makedirs(yolo_val_images, exist_ok=True)
            for img_file in os.listdir(valid_images_dir):
                src_path = os.path.join(valid_images_dir, img_file)
                dst_path = os.path.join(yolo_val_images, img_file)
                shutil.copy2(src_path, dst_path)

            # Copy labels
            os.makedirs(yolo_val_labels, exist_ok=True)
            for label_file in os.listdir(valid_labels_dir):
                src_path = os.path.join(valid_labels_dir, label_file)
                dst_path = os.path.join(yolo_val_labels, label_file)
                shutil.copy2(src_path, dst_path)

            print(f"  Validation data copied as 'val': {len(os.listdir(yolo_val_images))} images, {len(os.listdir(yolo_val_labels))} labels")

    # 4. Copy test folder (if exists)
    if has_test:
        print(f"Organizing test data...")
        # If folder doesn't exist or source has files, copy them
        if not os.path.exists(yolo_test_images) or len(os.listdir(test_images_dir)) > 0:
            # Copy images
            os.makedirs(yolo_test_images, exist_ok=True)
            for img_file in os.listdir(test_images_dir):
                src_path = os.path.join(test_images_dir, img_file)
                dst_path = os.path.join(yolo_test_images, img_file)
                shutil.copy2(src_path, dst_path)

            # Copy labels
            os.makedirs(yolo_test_labels, exist_ok=True)
            for label_file in os.listdir(test_labels_dir):
                src_path = os.path.join(test_labels_dir, label_file)
                dst_path = os.path.join(yolo_test_labels, label_file)
                shutil.copy2(src_path, dst_path)

            print(f"  Test data copied: {len(os.listdir(yolo_test_images))} images, {len(os.listdir(yolo_test_labels))} labels")

    # 5. Report updated structure
    print(f"\nUpdated structure:")
    for root, dirs, files in os.walk(dataset_dir):
        # Only show images and labels folders
        if 'images' in root or 'labels' in root:
            level = root.replace(dataset_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            if level <= 2:  # Only list first 3 levels
                sub_dirs = [d for d in dirs if d in ['train', 'val', 'test']]
                for d in sub_dirs:
                    sub_path = os.path.join(root, d)
                    file_count = len([f for f in os.listdir(sub_path) if os.path.isfile(os.path.join(sub_path, f))])
                    print(f"{indent}    {d}/ ({file_count} files)")

def update_dataset_yaml(dataset_dir):
    """Read data.yaml from downloaded dataset and reconfigure it"""
    source_yaml = os.path.join(dataset_dir, 'data.yaml')
    target_yaml = 'dataset.yaml'

    try:
        # Read original YAML
        with open(source_yaml, 'r') as f:
            data = yaml.safe_load(f)

        # Preserve class information
        class_names = data.get('names', [])
        nc = data.get('nc', len(class_names))

        # Create new configuration
        updated_data = {
            'path': os.path.abspath(dataset_dir),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test' if os.path.exists(os.path.join(dataset_dir, 'images/test')) else '',
            'nc': nc,
            'names': class_names
        }

        # Preserve other metadata if exists
        if 'roboflow' in data:
            updated_data['roboflow'] = data['roboflow']

        # Save new YAML
        with open(target_yaml, 'w') as f:
            yaml.dump(updated_data, f, sort_keys=False)

        print(f'Dataset configuration updated: {target_yaml}')
        print(f'Classes: {updated_data["names"]}')

        # For safety, show the updated dataset configuration
        print(f'Updated configuration:')
        for key, value in updated_data.items():
            print(f"  {key}: {value}")

        return True
    except Exception as e:
        print(f'YAML update error: {e}')
        return False

def analyze_dataset(dataset_dir):
    """Analyze dataset and provide statistics"""
    print(f"\n===== Dataset Analysis =====")
    
    try:
        # Get image and label counts
        train_img_dir = os.path.join(dataset_dir, 'images', 'train')
        val_img_dir = os.path.join(dataset_dir, 'images', 'val')
        test_img_dir = os.path.join(dataset_dir, 'images', 'test')
        
        train_label_dir = os.path.join(dataset_dir, 'labels', 'train')
        val_label_dir = os.path.join(dataset_dir, 'labels', 'val')
        test_label_dir = os.path.join(dataset_dir, 'labels', 'test')
        
        # Count files if directories exist
        train_img_count = len(os.listdir(train_img_dir)) if os.path.exists(train_img_dir) else 0
        val_img_count = len(os.listdir(val_img_dir)) if os.path.exists(val_img_dir) else 0
        test_img_count = len(os.listdir(test_img_dir)) if os.path.exists(test_img_dir) else 0
        
        train_label_count = len(os.listdir(train_label_dir)) if os.path.exists(train_label_dir) else 0
        val_label_count = len(os.listdir(val_label_dir)) if os.path.exists(val_label_dir) else 0
        test_label_count = len(os.listdir(test_label_dir)) if os.path.exists(test_label_dir) else 0
        
        # Print statistics
        print(f"Training set: {train_img_count} images, {train_label_count} labels")
        print(f"Validation set: {val_img_count} images, {val_label_count} labels")
        print(f"Test set: {test_img_count} images, {test_label_count} labels")
        print(f"Total images: {train_img_count + val_img_count + test_img_count}")
        
        # Check for label class distribution
        if os.path.exists('dataset.yaml'):
            with open('dataset.yaml', 'r') as f:
                data = yaml.safe_load(f)
            
            class_names = data.get('names', [])
            print(f"\nClass names: {class_names}")
            
            # Count instances of each class in the training set
            if os.path.exists(train_label_dir) and class_names:
                class_counts = {name: 0 for name in class_names}
                
                # Sample up to 50 label files to get class distribution
                sample_count = min(50, train_label_count)
                sample_files = os.listdir(train_label_dir)[:sample_count]
                
                for label_file in sample_files:
                    file_path = os.path.join(train_label_dir, label_file)
                    with open(file_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts and len(parts) >= 5:  # YOLO format: class x y w h
                                try:
                                    class_idx = int(parts[0])
                                    if 0 <= class_idx < len(class_names):
                                        class_counts[class_names[class_idx]] += 1
                                except (ValueError, IndexError):
                                    pass
                
                print("\nClass distribution (based on sample):")
                for name, count in class_counts.items():
                    print(f"  {name}: {count}")
                    
        return {
            'train_count': train_img_count,
            'val_count': val_img_count,
            'test_count': test_img_count,
            'total_count': train_img_count + val_img_count + test_img_count
        }
    except Exception as e:
        print(f"Dataset analysis error: {e}")
        return None

def check_dataset_integrity(dataset_dir):
    """Check dataset integrity - ensure all images have corresponding labels"""
    print("\n===== Dataset Integrity Check =====")
    
    issues_found = 0
    
    try:
        # Check train set
        train_img_dir = os.path.join(dataset_dir, 'images', 'train')
        train_label_dir = os.path.join(dataset_dir, 'labels', 'train')
        
        if os.path.exists(train_img_dir) and os.path.exists(train_label_dir):
            train_images = {os.path.splitext(f)[0] for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))}
            train_labels = {os.path.splitext(f)[0] for f in os.listdir(train_label_dir) if f.endswith('.txt')}
            
            # Find images without labels
            img_without_label = train_images - train_labels
            if img_without_label:
                print(f"Found {len(img_without_label)} training images without labels")
                if len(img_without_label) <= 5:  # Show the first few only
                    print(f"  Missing labels for: {', '.join(list(img_without_label))}")
                issues_found += len(img_without_label)
            
            # Find labels without images
            label_without_img = train_labels - train_images
            if label_without_img:
                print(f"Found {len(label_without_img)} training labels without images")
                if len(label_without_img) <= 5:  # Show the first few only
                    print(f"  Extra labels for: {', '.join(list(label_without_img))}")
                issues_found += len(label_without_img)
        
        # Check validation set
        val_img_dir = os.path.join(dataset_dir, 'images', 'val')
        val_label_dir = os.path.join(dataset_dir, 'labels', 'val')
        
        if os.path.exists(val_img_dir) and os.path.exists(val_label_dir):
            val_images = {os.path.splitext(f)[0] for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))}
            val_labels = {os.path.splitext(f)[0] for f in os.listdir(val_label_dir) if f.endswith('.txt')}
            
            # Find images without labels
            img_without_label = val_images - val_labels
            if img_without_label:
                print(f"Found {len(img_without_label)} validation images without labels")
                issues_found += len(img_without_label)
            
            # Find labels without images
            label_without_img = val_labels - val_images
            if label_without_img:
                print(f"Found {len(label_without_img)} validation labels without images")
                issues_found += len(label_without_img)
        
        if issues_found == 0:
            print("No dataset integrity issues found!")
        else:
            print(f"Total issues found: {issues_found}")
            print("Consider fixing these issues for better training results.")
        
        return issues_found
    except Exception as e:
        print(f"Dataset integrity check error: {e}")
        return -1

if __name__ == "__main__":
    # If run directly, test functionality
    test_dir = "datasets/test_dataset"
    os.makedirs(test_dir, exist_ok=True)
    print("This module provides dataset management functions.")
    print("To test, provide a Roboflow URL as argument.")