#!/usr/bin/env python3
# __init__.py - Package initialization for YOLO11 Training Framework

"""
YOLO11 Training Framework for Agricultural Applications

This package provides a modular approach to training YOLO11 object detection models,
optimized for Google Colab and agricultural applications.
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

# Export main components
from .setup_utils import check_gpu, install_required_packages, mount_drive, check_environment
from .hyperparameters import create_hyperparameters_file, load_hyperparameters
from .dataset_utils import download_dataset, fix_directory_structure, update_dataset_yaml
from .memory_utils import show_memory_usage, clean_memory, run_training_with_memory_cleanup
from .training import train_model, save_to_drive, validate_model, export_model
from .config import load_config, save_config, get_options_from_config

# For convenience, also export the main function
from .main import main, interactive_setup
