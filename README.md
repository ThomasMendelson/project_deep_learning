# Deep Learning for 3D Cell Segmentation and Tracking

This repository contains the code for our deep learning project on 3D cell segmentation and tracking, inspired by the [Cell Tracking Challenge](https://celltrackingchallenge.net/). The goal of this project is to leverage advanced deep learning models to accurately segment and track cells in challenging microscopy images, specifically using the Fluo-N3DH-SIM+ dataset.

## Challenge Overview
The [Cell Tracking Challenge](https://celltrackingchallenge.net/) provides a set of benchmark datasets designed to evaluate the performance of segmentation and tracking algorithms in microscopy images. Our project focuses on the 3D datasets, where the main challenge is accurately identifying and tracking individual cell nuclei in densely packed 3D fluorescence microscopy images.

## Dataset
We used the [Fluo-N3DH-SIM+ dataset](https://celltrackingchallenge.net/3d-datasets/) from the Cell Tracking Challenge. This dataset consists of 3D time-lapse fluorescence microscopy images, with ground truth annotations for segmentation and tracking. The dataset is divided into separate folders for training and validation:
- **01 (Training Images):** Contains the raw 3D image volumes for training.
- **01_GT (Training Ground Truth):** Contains segmentation (`SEG`) and tracking (`TRA`) annotations.
- **02 (Validation Images):** Contains the raw 3D image volumes for validation.
- **02_GT (Validation Ground Truth):** Contains segmentation and tracking annotations for validation.

## Model Architecture
Our architecture is inspired by the UNETR model proposed in the paper *"UNETR: Transformers for 3D Medical Image Segmentation"* by Hatamizadeh et al. The model combines Vision Transformers (ViTs) with a U-Net-like structure to leverage both global context and local spatial information, making it well-suited for the complex task of cell segmentation in 3D images.

![Architecture Diagram](architecture%20image.png)


The key components of our architecture include:
- **Vision Transformer:** Processes the input image using multi-head self-attention to capture long-range dependencies.
- **Skip Connections:** Connect encoder and decoder at multiple resolutions to preserve spatial details.
- **U-Net Decoder:** Uses the outputs of the Vit and skip connections to form a U-Net decoder stracture. 
- **Dual Output Layers:** Provides separate outputs for segmentation and marker maps, aiding in precise cell identification and boundary delineation.

### Citation
The model architecture is based on:
Hatamizadeh, A., et al. (2021). *UNETR: Transformers for 3D Medical Image Segmentation*. Available: https://arxiv.org/abs/2103.10504

## Directory Structure
- **models/**: Contains the code for the implemented models.
- **dataset/** and **dataset3D/** and **utils.py**: Include functions for loading and preprocessing data, as well as utilities for creating data loaders.
- **train3D.py, train_swin.py, train_vit2.py**: Scripts for training different models. Each script allows training with specific configurations suited to the corresponding architecture.
- **main.py**: The main script where you can select which model to train and define key parameters.

## Usage
1. **Download the Dataset:**
   - Download the dataset from the [Fluo-N3DH-SIM+ dataset page](https://celltrackingchallenge.net/3d-datasets/).
   - Update the file paths in the configuration files and scripts to match the downloaded dataset location.

2. **Download the requirements:**
   ```bash
   pip install -r requirements.txt

3. **Configure Hyperparameters:**
   - Adjust the hyperparameters in the scripts (`train3D.py`, `train_swin.py`, `train_vit2.py`) to suit your experimental setup, including learning rates, batch sizes, and loss function weights.

4. **Train the Model:**
   - Use the `main.py` to choose which model to train and set additional options for your experiments.
   - Example command:
     ```bash
     python main.py --model vit_unet
     ```

5. **Evaluate and Analyze:**
   - After training, use the saved models to perform evaluation with **check_accuray** function in **utils.py** and generate segmentation results. 

## Additional Notes
- Ensure all dependencies are installed, preferably in a virtual environment or using Docker for consistency.
- For hyperparameter tuning, especially regarding loss function weights and regularization settings, refer to the ablation studies conducted in our experiments.

## Contact
For any issues or questions, please feel free to contact [Thomas Mendelson](https://github.com/ThomasMendelson).

