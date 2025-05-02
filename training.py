#!/usr/bin/env python3
# training.py - Model training functions for YOLO11

import os
import sys
import yaml
import torch
import shutil
from pathlib import Path
from ultralytics import YOLO

from memory_utils import show_memory_usage, clean_memory, run_training_with_memory_cleanup

def train_model(options, hyp=None, resume=False, epochs=None):
    """Train YOLO11 model"""
    # Model selection
    model_path = options['model']

    print(f"Loading model: {model_path}")
    # Check model path - YOLO will download automatically if file doesn't exist
    if not os.path.exists(model_path) and model_path.startswith('yolo11') and model_path.endswith('.pt'):
        print(f"Model file not found. YOLO will automatically download the '{model_path}' model...")

    # Load pre-trained model or create a new one
    if resume:
        # Find latest checkpoint
        runs_dir = Path(options.get('save_dir', '') or options.get('project', 'runs/train'))
        exp_name = options.get('name', 'exp')
        weights_dir = runs_dir / exp_name / 'weights'

        if weights_dir.exists():
            last_pt = weights_dir / 'last.pt'
            if last_pt.exists():
                model_path = str(last_pt)
                print(f'Resuming from: {model_path}')
            else:
                print('Last checkpoint not found, starting from scratch')
                resume = False
        else:
            print('Previous training not found, starting from scratch')
            resume = False

    # Set epoch save interval
    save_interval = int(input("\nHow often to save the model (epochs)? (default: 50): ") or "50")

    # Define custom callback function for periodic saving
    def save_periodic_model(trainer):
        current_epoch = trainer.epoch + 1  # Epoch count starts from 0 so add 1
        if current_epoch % save_interval == 0:
            # Set path for periodic save
            periodic_save_path = Path(trainer.save_dir) / "weights" / f"epoch_{current_epoch}.pt"
            trainer.model.save(periodic_save_path)
            print(f"\n--- Model saved for epoch {current_epoch}: {periodic_save_path} ---")

            # Also copy to Google Drive (if running in Colab)
            try:
                if 'google.colab' in sys.modules:
                    drive_folder = "/content/drive/MyDrive/Tarim/YapayZeka_model/Model/YOLO11_Egitim"
                    os.makedirs(drive_folder, exist_ok=True)
                    drive_path = f"{drive_folder}/epoch_{current_epoch}.pt"
                    import shutil
                    shutil.copy(str(periodic_save_path), drive_path)
                    print(f"Epoch {current_epoch} model also saved to Drive: {drive_path}")
            except Exception as e:
                print(f"Error copying to Drive: {e}")

    try:
        # Load model
        #model = YOLO(model_path)
        #print(f"Model loaded successfully: {model_path}")
        # Şu satırdan önce (68. satır civarı):
        model = YOLO(model_path)
        
        # Aşağıdaki kontrol kodunu ekleyin:
        # Model dosya yolu kontrolü - Colab'daki yol sorununu düzelt
        if 'google.colab' in sys.modules:
            # Eğer model yolu belirlediğimiz klasöre aitse
            if model_path.startswith('yolo11') and model_path.endswith('.pt'):
                # Doğrudan /content altına bak
                if os.path.exists(f"/content/{model_path}"):
                    model_path = f"/content/{model_path}"
                elif os.path.exists(f"/content/yolo11_models/{model_path}"):
                    model_path = f"/content/yolo11_models/{model_path}"
            print(f"Kontrol edilen model yolu: {model_path}")
    except Exception as e:
        print(f"Model loading error: {e}")
        print("Trying alternative model...")
        # Fall back to yolo11l.pt model
        model = YOLO('yolo11l.pt')
        print("Using yolo11l.pt model.")
        model_path = 'yolo11l.pt'
        options['model'] = model_path

    # Settings for periodic memory cleanup
    cleanup_frequency = int(input("\nRAM cleanup frequency (clean every N epochs? e.g., 10): ") or "10")
    use_cache = input("\nCache dataset? (y/n) (default: y): ").lower() or "y"
    use_cache = use_cache.startswith("y")

    # Set training parameters
    train_args = {
        'data': options['data'],
        'epochs': epochs if epochs is not None else options['epochs'],
        'imgsz': options['imgsz'],
        'batch': options['batch'],
        'project': options.get('project', 'runs/train'),
        'name': options.get('name', 'exp'),
        'device': options.get('device', 'cpu'),  # Default to CPU
        'workers': options.get('workers', 8),
        'exist_ok': options.get('exist_ok', False),
        'pretrained': options.get('pretrained', True),
        'optimizer': options.get('optimizer', 'auto'),
        'verbose': options.get('verbose', True),
        'seed': options.get('seed', 0),
        'cache': use_cache,  # Cache dataset - improves training performance
        'resume': resume,  # Resume from checkpoint
        'callbacks': [save_periodic_model],  # Add custom callback
    }

    # Add hyperparameters if available (as fixed constants, not as hyp.yaml file!)
    if hyp is not None and options.get('use_hyp', True):
        # Get patience value from hyperparameters
        if 'patience' in hyp:
            train_args['patience'] = hyp['patience']

        # Add lr0 (initial learning rate) and other supported custom parameters
        if 'lr0' in hyp:
            train_args['lr0'] = hyp['lr0']

        # Add other parameters directly supported by YOLO11 training
        supported_params = ['lrf', 'warmup_epochs', 'warmup_momentum', 'box', 'cls']
        for param in supported_params:
            if param in hyp:
                train_args[param] = hyp[param]

        print(f"Compatible settings transferred from hyperparameters")
    else:
        # Default patience value
        train_args['patience'] = 50

    print('Training parameters:')
    for k, v in train_args.items():
        print(f'  {k}: {v}')

    # Show memory status before training
    show_memory_usage("Training Start Memory Status")

    # Manage model training with periodic memory cleanup
    results = run_training_with_memory_cleanup(model, train_args, cleanup_frequency)

    # Clean memory after training
    clean_memory()

    return results

def save_to_drive(options, results=None):
    """Save trained model to Google Drive (for Colab)"""
    try:
        # Connect to Google Drive
        from google.colab import drive
        print("\nConnecting to Google Drive...")
        drive.mount('/content/drive')

        # Determine model file paths
        project_dir = options.get('project', 'runs/train')
        exp_name = options.get('name', 'exp')
        best_model_path = f"{project_dir}/{exp_name}/weights/best.pt"
        last_model_path = f"{project_dir}/{exp_name}/weights/last.pt"

        # Create folder in Drive
        drive_folder = "/content/drive/MyDrive/Tarim/YapayZeka_model/Model/YOLO11_Egitim"
        os.makedirs(drive_folder, exist_ok=True)

        # Copy models
        import shutil
        if os.path.exists(best_model_path):
            drive_best_path = f"{drive_folder}/best_model.pt"
            shutil.copy(best_model_path, drive_best_path)
            print(f"Best model saved to Drive: {drive_best_path}")

        if os.path.exists(last_model_path):
            drive_last_path = f"{drive_folder}/last_model.pt"
            shutil.copy(last_model_path, drive_last_path)
            print(f"Last model saved to Drive: {drive_last_path}")

        # Also copy training results
        results_path = f"{project_dir}/{exp_name}/results.csv"
        if os.path.exists(results_path):
            drive_results_path = f"{drive_folder}/results.csv"
            shutil.copy(results_path, drive_results_path)
            print(f"Training results saved to Drive: {drive_results_path}")

        print(f"All files saved to {drive_folder}.")
        return True
    except Exception as e:
        print(f"Error saving to Google Drive: {e}")
        print("Don't forget to save your model files manually!")
        return False

def validate_model(model_path, data_yaml, batch_size=16, img_size=640):
    """Validate a trained model on test/validation data"""
    try:
        print(f"\n===== Model Validation =====")
        print(f"Loading model: {model_path}")
        
        # Load the model
        model = YOLO(model_path)
        
        # Run validation
        print(f"Running validation on: {data_yaml}")
        results = model.val(
            data=data_yaml,
            batch=batch_size,
            imgsz=img_size,
            verbose=True
        )
        
        # Report results
        print("\nValidation Results:")
        metrics = ["precision", "recall", "mAP50", "mAP50-95"]
        for metric in metrics:
            if hasattr(results, metric):
                value = getattr(results, metric)
                print(f"  {metric}: {value:.4f}")
        
        return results
    except Exception as e:
        print(f"Validation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def export_model(model_path, format='onnx', img_size=640, simplify=True):
    """Export YOLO model to different formats"""
    try:
        print(f"\n===== Model Export =====")
        print(f"Loading model: {model_path}")
        
        # Load the model
        model = YOLO(model_path)
        
        # Available formats
        formats = ['torchscript', 'onnx', 'openvino', 'engine', 'coreml', 'saved_model', 
                  'pb', 'tflite', 'edgetpu', 'tfjs', 'paddle', 'ncnn']
        
        if format.lower() not in formats:
            print(f"Unsupported format: {format}")
            print(f"Supported formats: {', '.join(formats)}")
            return None
        
        # Export the model
        print(f"Exporting to {format} format...")
        model.export(
            format=format, 
            imgsz=img_size,
            simplify=simplify
        )
        
        print(f"Model exported successfully!")
        
        # Find exported file
        model_dir = os.path.dirname(model_path)
        if not model_dir:
            model_dir = '.'
            
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        exported_file = None
        
        # Map of format to extension
        format_extensions = {
            'torchscript': '.torchscript',
            'onnx': '.onnx',
            'openvino': '_openvino_model',
            'engine': '.engine',
            'coreml': '.mlmodel',
            'saved_model': '_saved_model',
            'pb': '.pb',
            'tflite': '.tflite',
            'edgetpu': '_edgetpu.tflite',
            'tfjs': '_web_model',
            'paddle': '_paddle_model',
            'ncnn': '_ncnn_model'
        }
        
        # Look for exported file
        if format in format_extensions:
            extension = format_extensions[format]
            potential_file = os.path.join(model_dir, base_name + extension)
            if os.path.exists(potential_file):
                exported_file = potential_file
            elif os.path.isdir(potential_file):
                exported_file = potential_file
                
        if exported_file:
            print(f"Exported file: {exported_file}")
            
        return exported_file
    except Exception as e:
        print(f"Export error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("This module provides model training functions.")
    print("It is not meant to be run directly.")
