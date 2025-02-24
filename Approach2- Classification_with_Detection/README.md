Detection and Classification of Bacterial Cells
================================================

Overview
--------
This module implements the detection and classification pipeline described in the paper 
"Deep Learning Imaging Analysis to Identify Bacterial Metabolic States Associated with Carcinogen Production." 
In this approach, individual cells are first detected in microscopy images using adaptive (or OTSU) thresholding.
The detected cells are then classified using a pre-trained ResNet model to determine the metabolic state of 
Clostridium scindens under various conditions.

Pipeline Workflow
-----------------
1. **Cell Detection:** (adaptive_thresholding.py)
   - **Method:** Adaptive thresholding (or OTSU thresholding) is used to segment cells from the background.
   - **Output:** Individual cell regions are cropped and saved for further analysis.
   
2. **Cell Classification:** (ResNet.py)
   - **Method:** A pre-trained ResNet model is used to classify each detected cell into its corresponding 
     metabolic state.
   - **Output:** Classification results along with performance metrics (accuracy, confusion matrices, etc.) are generated.

Installation and Requirements
-----------------------------
- **Python Version:** Python 3.x
- **Dependencies:**  
  - OpenCV  
  - NumPy  
  - PyTorch  
  - scikit-image  
  - Other dependencies as listed in `requirements.txt`
  
To install the required packages, run: pip install -r requirements.txt






