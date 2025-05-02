#!/usr/bin/env python3
# config.py - Configuration management for YOLO11 training

import os
import yaml
import json
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    # Model settings
    'model': 'yolo11m.pt',
    'imgsz': 640,
    'batch': 16,
    'epochs': 500,
    
    # Dataset settings
    'data': 'dataset.yaml',
    'roboflow_url': '',
    
    # Training settings
    'device': 'auto',
    'workers': 8,
    'optimizer': 'auto',
    'patience': 50,
    'save_interval': 50,
    'cleanup_frequency': 10,
    'use_cache': True,
    
    # Output settings
    'project': 'runs/train',
    'name': 'exp',
    'exist_ok': True,
    'pretrained': True,
    'verbose': True,
    'resume': False,
    
    # Hyperparameters
    'use_hyp': True,
    'lr0': 0.01,
    'weight_decay': 0.001,
    'warmup_epochs': 3.0,
    'mosaic': 1.0,
    'fliplr': 0.5
}

def save_config(config, file_path='config.yaml'):
    """Save configuration to a YAML file"""
    try:
        with open(file_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        print(f"Configuration saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False

def load_config(file_path='config.yaml'):
    """Load configuration from a YAML file"""
    try:
        if not os.path.exists(file_path):
            print(f"Configuration file {file_path} not found. Using defaults.")
            return DEFAULT_CONFIG.copy()
            
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {file_path}")
        
        # Merge with defaults to ensure all keys exist
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)
        
        return merged_config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Using default configuration")
        return DEFAULT_CONFIG.copy()

def create_config_template(file_path='config_template.yaml'):
    """Create a template configuration file with comments"""
    template = """# YOLO11 Training Configuration Template

#-------------------------
# Model Settings
#-------------------------
model: yolo11m.pt  # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
imgsz: 640         # Image size (must be multiple of 32)
batch: 16          # Batch size (reduce for less memory usage)
epochs: 500        # Number of training epochs

#-------------------------
# Dataset Settings
#-------------------------
data: dataset.yaml  # Path to dataset YAML file
roboflow_url: ''    # Roboflow dataset URL for downloading

#-------------------------
# Training Settings
#-------------------------
device: auto        # Device to use (auto, 0, 1, cpu)
workers: 8          # Number of worker threads for data loading
optimizer: auto     # Optimizer to use (auto, SGD, Adam, AdamW)
patience: 50        # Patience for early stopping
save_interval: 50   # Save model every N epochs
cleanup_frequency: 10  # Clean memory every N epochs
use_cache: true     # Cache dataset in memory for faster training

#-------------------------
# Output Settings
#-------------------------
project: runs/train  # Directory to save results
name: exp            # Experiment name
exist_ok: true       # Overwrite existing experiment
pretrained: true     # Use pretrained weights
verbose: true        # Verbose output
resume: false        # Resume training from last checkpoint

#-------------------------
# Hyperparameters
#-------------------------
use_hyp: true        # Use hyperparameter file
lr0: 0.01            # Initial learning rate
weight_decay: 0.001  # Weight decay
warmup_epochs: 3.0   # Warmup epochs
mosaic: 1.0          # Mosaic augmentation
fliplr: 0.5          # Horizontal flip augmentation
"""
    try:
        with open(file_path, 'w') as f:
            f.write(template)
        print(f"Configuration template created at {file_path}")
        return True
    except Exception as e:
        print(f"Error creating configuration template: {e}")
        return False

def get_config_from_options(options):
    """Convert interactive options to config dictionary"""
    config = DEFAULT_CONFIG.copy()
    
    # Update config with options
    for key, value in options.items():
        if key in config:
            config[key] = value
    
    return config

def get_options_from_config(config):
    """Convert config dictionary to options format expected by training functions"""
    options = {}
    
    # Copy relevant keys
    for key in ['model', 'imgsz', 'batch', 'epochs', 'data', 'roboflow_url', 
               'device', 'workers', 'project', 'name', 'exist_ok', 
               'pretrained', 'optimizer', 'verbose', 'resume', 'use_hyp']:
        if key in config:
            options[key] = config[key]
    
    return options

if __name__ == "__main__":
    # Create template config if run directly
    create_config_template()
    
    # Save default config
    save_config(DEFAULT_CONFIG, 'default_config.yaml')
    
    print("\nTo use this configuration module:")
    print("1. Modify the template config file")
    print("2. Load it in your main script:")
    print("   ```")
    print("   from config import load_config")
    print("   config = load_config('your_config.yaml')")
    print("   ```")
