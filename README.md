# YOLO11 Training Framework for Agricultural Applications

This framework provides a modular and organized approach to training YOLO11 object detection models, optimized for use in Google Colab. The code is structured to handle the entire workflow from dataset preparation to model training and export.

## Project Structure

The project has been reorganized into the following components:

- **main.py**: The main script that orchestrates the entire training process
- **setup_utils.py**: Setup utilities including GPU detection, package installation
- **hyperparameters.py**: Hyperparameter management for optimizing model training
- **dataset_utils.py**: Dataset download, preparation, and analysis
- **memory_utils.py**: Memory management to optimize performance
- **training.py**: Model training, validation, and export functions
- **model_downloader.py**: Utility for downloading YOLO11 model variants
- **config.py**: Configuration management for reproducible experiments

## Installation and Requirements

1. This framework is optimized for Google Colab but can be used in any Python environment
2. Required packages: `ultralytics`, `pyyaml`, `torch`, `psutil`, `requests`
3. For best performance, use a GPU-enabled environment

## Usage

1. Run the main script:
   ```
   python main.py
   ```

2. Choose between downloading models or training setup

3. If downloading models:
   - Select individual models or download all types
   - Models will be saved to the specified directory

4. For training:
   - Follow the interactive prompts to configure your training
   - The script will download your dataset from Roboflow
   - Set up the proper directory structure
   - Configure hyperparameters
   - Train the model with memory optimization
   - Save the model to Google Drive (if in Colab)

## Downloading YOLO11 Models

The framework includes a dedicated module for downloading official YOLO11 models:

```python
# Download a specific model
from model_downloader import download_specific_model_type
model_path = download_specific_model_type(
    model_type="detection",  # Options: detection, segmentation, classification, pose, obb
    size="m",                # Options: s, m, l, x
    save_dir="models"        # Directory to save the model
)

# Or download multiple models
from model_downloader import download_yolo11_models
models = download_yolo11_models(
    save_dir="models",                # Directory to save models
    selected_models=["yolo11m.pt"]    # List of models to download (or None for all)
)
```

Available model types:
- **Detection**: Basic object detection models (e.g., yolo11m.pt)
- **Segmentation**: Instance segmentation models (e.g., yolo11m-seg.pt)
- **Classification**: Image classification models (e.g., yolo11m-cls.pt)
- **Pose**: Pose estimation models (e.g., yolo11m-pose.pt)
- **OBB**: Oriented bounding box models (e.g., yolo11m-obb.pt)

## Key Features

- **Modular Design**: Each aspect of the training process is separated into its own module
- **Memory Management**: Periodic memory cleanup to avoid out-of-memory errors
- **Hyperparameter Optimization**: Customizable training parameters
- **Google Drive Integration**: Automatic saving of models to Drive
- **Automatic GPU Detection**: Uses GPU if available, falls back to CPU if not
- **Dataset Integrity Checking**: Ensures your dataset is properly formatted
- **Model Management**: Easy download of official YOLO11 model variants

## Modules Overview

### main.py
The main entry point that orchestrates the entire training process and model downloading.

### setup_utils.py
Handles environment setup including GPU detection, package installation, and directory creation.

### hyperparameters.py
Manages training hyperparameters for optimizing model performance.

### dataset_utils.py
Handles downloading datasets from Roboflow, organizing the directory structure, and verifying dataset integrity.

### memory_utils.py
Provides utilities for monitoring and optimizing memory usage during training.

### training.py
Contains functions for model training, validation, and exporting trained models to different formats.

### model_downloader.py
Utility for downloading official YOLO11 model variants from the Ultralytics GitHub repository.

### config.py
Manages configuration files for reproducible experiments.

## Best Practices

1. **Start Small**: Begin with a smaller model (yolo11n or yolo11s) for initial testing
2. **Monitor Memory**: Keep an eye on memory usage, especially on free Colab instances
3. **Save Regularly**: Models are saved periodically to prevent data loss
4. **Hyperparameter Tuning**: Adjust hyperparameters based on your specific dataset
5. **Download Models in Advance**: For better reliability, download models in advance rather than during training

## Example Workflow

```python
# Import the framework
from main import main

# Run the interactive setup and training
if __name__ == "__main__":
    main()
```

## License

This project is provided for educational and research purposes.

## Acknowledgements

- Ultralytics for the YOLO11 implementation
- Roboflow for dataset management capabilities
