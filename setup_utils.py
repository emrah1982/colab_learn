#!/usr/bin/env python3
# setup_utils.py - Setup utilities for YOLO11 training

import os
import sys
import subprocess
import torch

def install_required_packages(required_packages):
    """Check and install required packages if not available"""
    for package in required_packages:
        try:
            __import__(package)
            print(f"{package} is already installed.")
        except ImportError:
            print(f"{package} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} successfully installed!")

def check_gpu():
    """Check GPU status and return appropriate device setting"""
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        print(f"GPU found: {gpu_name} (Total: {gpu_count})")
        return "auto"  # Use auto if GPU is available
    else:
        print("No GPU found. Running in CPU mode.")
        print("For faster training in Colab: Runtime > Change runtime type > T4 GPU")
        return "cpu"  # Use CPU if no GPU is available

def setup_base_dirs(base_path="/content/drive/MyDrive/Tarim"):
    """Set up base directories for the project"""
    # Create directory structure if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Create subdirectories
    model_dir = os.path.join(base_path, "YapayZeka_model/Model/YOLO11_Egitim")
    data_dir = os.path.join(base_path, "Data")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Base directories created:")
    print(f"- Model directory: {model_dir}")
    print(f"- Data directory: {data_dir}")
    
    return {
        "base_path": base_path,
        "model_dir": model_dir,
        "data_dir": data_dir
    }

def mount_drive():
    """Mount Google Drive in Colab"""
    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
        print("Google Drive mounted successfully.")
        return True
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        print("You may need to mount it manually.")
        return False

def check_environment():
    """Check the environment and report system information"""
    # Python version
    python_version = sys.version
    print(f"Python version: {python_version}")
    
    # PyTorch version
    torch_version = torch.__version__
    print(f"PyTorch version: {torch_version}")
    
    # CUDA version if available
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        print(f"CUDA version: {cuda_version}")
        
        # GPU info
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # Convert to GB
            print(f"GPU {i}: {gpu_name} with {gpu_memory:.2f} GB memory")
    else:
        print("CUDA not available")
        
    # Check available RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        print(f"Total RAM: {ram_gb:.2f} GB")
    except ImportError:
        print("psutil not installed, can't check RAM")
        
    # Check if in Colab
    in_colab = 'google.colab' in sys.modules
    print(f"Running in Google Colab: {in_colab}")
    
    return {
        "python_version": python_version,
        "torch_version": torch_version,
        "in_colab": in_colab,
        "gpu_available": torch.cuda.is_available()
    }

if __name__ == "__main__":
    # If this file is run directly, perform a system check
    print("Running environment check...")
    check_environment()
    check_gpu()
