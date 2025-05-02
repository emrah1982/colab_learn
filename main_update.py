#!/usr/bin/env python3
# main.py - YOLO11 Training Main Script for Colab

import os
import sys
from pathlib import Path

# Import components
from setup_utils import check_gpu, install_required_packages
from hyperparameters import create_hyperparameters_file, load_hyperparameters
from dataset_utils import download_dataset, fix_directory_structure, update_dataset_yaml
from memory_utils import run_training_with_memory_cleanup, show_memory_usage, clean_memory
from training import train_model, save_to_drive
from model_downloader import download_yolo11_models, download_specific_model_type

# Check if running in Colab
def is_colab():
    """Check if running in Google Colab"""
    try:
        import google.colab
        print("Google Colab environment detected.")
        return True
    except:
        print("Running in local environment.")
        return False

def download_models_menu():
    """Interactive menu for downloading YOLO11 models"""
    print("\n===== YOLO11 Model Download =====")
    
    # Ask for save directory
    default_dir = os.path.join(os.getcwd(), "yolo11_models")
    save_dir = input(f"\nSave directory (default: {default_dir}): ") or default_dir
    
    # Ask for download type
    print("\nDownload options:")
    print("1. Download single model")
    print("2. Download all detection models")
    print("3. Download all models (all types)")
    
    choice = input("\nYour choice (1-3): ")
    
    if choice == "1":
        # Single model download
        print("\nSelect model type:")
        print("1. Detection (default)")
        print("2. Segmentation")
        print("3. Classification")
        print("4. Pose")
        print("5. OBB (Oriented Bounding Box)")
        
        model_type_map = {
            "1": "detection",
            "2": "segmentation",
            "3": "classification",
            "4": "pose",
            "5": "obb"
        }
        
        model_type_choice = input("Enter choice (1-5, default: 1): ") or "1"
        model_type = model_type_map.get(model_type_choice, "detection")
        
        # Ask for model size
        print("\nSelect model size:")
        print("1. Small (s)")
        print("2. Medium (m) (default)")
        print("3. Large (l)")
        print("4. Extra Large (x)")
        
        size_map = {
            "1": "s",
            "2": "m",
            "3": "l",
            "4": "x"
        }
        
        size_choice = input("Enter choice (1-4, default: 2): ") or "2"
        size = size_map.get(size_choice, "m")
        
        # Download the selected model
        model_path = download_specific_model_type(model_type, size, save_dir)
        if model_path:
            print(f"\nModel downloaded successfully to: {model_path}")
    
    elif choice == "2":
        # Download all detection models
        detection_models = ["yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
        downloaded = download_yolo11_models(save_dir, detection_models)
        print(f"\nDownloaded {len(downloaded)} detection models to {save_dir}")
    
    elif choice == "3":
        # Download all models
        downloaded = download_yolo11_models(save_dir)
        print(f"\nDownloaded {len(downloaded)} models to {save_dir}")
    
    else:
        print("\nInvalid choice. No models downloaded.")
        return None
    
    return save_dir

def interactive_setup():
    """Interactively collect training parameters from the user"""
    print("\n===== YOLO11 Training Setup =====")

    # Ask for Roboflow URL
    roboflow_url = input("\nRoboflow URL (https://universe.roboflow.com/ds/...): ").strip()

    # Ask for number of epochs
    while True:
        try:
            epochs = int(input("\nNumber of epochs [100-5000 recommended] (default: 500): ") or "500")
            if epochs <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    # Ask for model size
    print("\nSelect model size:")
    print("1) yolo11n.pt - Nano (fastest, lowest accuracy)")
    print("2) yolo11s.pt - Small (fast, medium accuracy)")
    print("3) yolo11m.pt - Medium (balanced)")
    print("4) yolo11l.pt - Large (high accuracy, slow)")
    print("5) yolo11x.pt - XLarge (highest accuracy, slowest)")

    while True:
        model_choice = input("\nYour choice [1-5] (default: 3): ") or "3"
        model_options = {
            "1": "yolo11n.pt",
            "2": "yolo11s.pt",
            "3": "yolo11m.pt",
            "4": "yolo11l.pt",
            "5": "yolo11x.pt"
        }
        if model_choice in model_options:
            model = model_options[model_choice]
            
            # Check if model exists, offer to download if not
            model_dir = os.path.join(os.getcwd(), "yolo11_models")
            model_path = os.path.join(model_dir, model)
            
            if not os.path.exists(model_path):
                print(f"\nModel {model} not found locally.")
                download_now = input("Would you like to download it now? (y/n, default: y): ").lower() or "y"
                
                if download_now.startswith("y"):
                    os.makedirs(model_dir, exist_ok=True)
                    download_specific_model_type("detection", model[6], model_dir)
                else:
                    print(f"Model will be automatically downloaded during training.")
            
            break
        print("Please select a number between 1-5.")

    # Ask for batch size
    while True:
        try:
            batch_size = int(input("\nBatch size (default: 16, 8 or 4 recommended for low RAM): ") or "16")
            if batch_size <= 0:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    # Ask for image size
    while True:
        try:
            img_size = int(input("\nImage size (default: 640, must be a multiple of 32): ") or "640")
            if img_size <= 0 or img_size % 32 != 0:
                print("Please enter a positive number that is a multiple of 32.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")

    # Option to use hyperparameter file
    use_hyp = input("\nUse hyperparameter file (hyp.yaml)? (y/n) (default: y): ").lower() or "y"
    use_hyp = use_hyp.startswith("y")

    # Automatically check GPU/CPU
    device = check_gpu()

    # Return parameters as options dictionary
    options = {
        'roboflow_url': roboflow_url,
        'epochs': epochs,
        'model': model,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,  # Automatically determined from GPU check
        'data': 'dataset.yaml',
        'project': 'runs/train',
        'name': 'exp',
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'exist_ok': True,
        'resume': False,
        'use_hyp': use_hyp
    }

    print("\n===== Selected Parameters =====")
    for key, value in options.items():
        print(f"{key}: {value}")

    confirm = input("\nContinue with these parameters? (y/n): ").lower()
    if confirm != 'y' and confirm != 'yes':
        print("Setup cancelled.")
        return None

    return options

def main():
    """Main function - optimized for Colab"""
    print("\n===== YOLO11 Training Framework =====")
    print("1. Download models")
    print("2. Training setup")
    print("3. Exit")
    
    choice = input("\nSelect an option (1-3): ")
    
    if choice == "1":
        # Download models
        download_models_menu()
        # After downloading, ask if they want to continue to training
        train_now = input("\nProceed to training setup? (y/n, default: y): ").lower() or "y"
        if not train_now.startswith("y"):
            return
        
    if choice == "1" or choice == "2":
        # Check environment
        in_colab = is_colab()
        
        # Install required packages
        install_required_packages(['ultralytics', 'pyyaml'])

        # Create or use existing hyperparameter file
        hyp_path = create_hyperparameters_file()
        hyperparameters = load_hyperparameters(hyp_path)

        # Interactive setup
        options = interactive_setup()
        if options is None:
            return

        # Download dataset if Roboflow URL is provided
        if 'roboflow_url' in options and options['roboflow_url']:
            if not download_dataset(options['roboflow_url']):
                print('Failed to download dataset. Exiting...')
                return
        else:
            print('Please enter a valid Roboflow URL.')
            return

        # Train the model
        results = train_model(options, hyp=hyperparameters, resume=options.get('resume', False), epochs=options['epochs'])

        if results:
            print('Training completed!')
            print(f'Results: {results}')
        else:
            print('Training failed or was interrupted.')

        # Save to Drive if in Colab
        if in_colab:
            save_to_drive(options, results)
    
    elif choice == "3":
        print("Exiting...")
    
    else:
        print("Invalid choice. Exiting...")

if __name__ == "__main__":
    try:
        # Call main function
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nProcess completed.")
