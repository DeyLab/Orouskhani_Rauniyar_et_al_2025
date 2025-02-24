Deep Learning Imaging Analysis to Identify Bacterial Metabolic States Associated with Carcinogen Production
=============================================================================================================

Overview
--------
This repository contains the code and resources accompanying our paper titled "Deep Learning Imaging Analysis to Identify Bacterial Metabolic States Associated with Carcinogen Production". Our work leverages deep learning techniques to analyze microscopy images in order to identify metabolic states of bacteria that are associated with carcinogen production.

Approaches
----------
We investigated three primary approaches in our study:

1. Whole Image Classification
   --------------------------------
   - **Models Used:** ResNet and DenseNet
   - **Description:** In this approach, entire microscopy images are fed into convolutional neural networks (CNNs) for classification. The networks are trained to differentiate between various bacterial metabolic states based on global image features.
   - **Implementation:** The provided scripts include data loading, model training, validation, and testing routines.

2. Cell Detection and Classification
   --------------------------------------
   - **Detection:** Adaptive thresholding techniques are employed to detect individual cells within microscopy images.
   - **Classification:** Once the cells are detected, pre-trained classification models are used to assess the metabolic state of each cell.
   - **Purpose:** This two-step process allows us to focus on cell-level features, which can improve the accuracy of metabolic state detection.
   - **Implementation:** Code is included for both the adaptive thresholding process and the integration with pre-trained classifiers.

3. Cell Segmentation
   --------------------
   - **Tool Used:** nnU-Net
   - **Description:** For a detailed cellular analysis, we apply nnU-Net for precise cell segmentation. This method isolates cells from the background, allowing for enhanced morphological assessment.
   - **Implementation:** The repository provides configuration files and training scripts necessary to run the segmentation model.

Usage Instructions
------------------
1. **Environment Setup:**  
   - Ensure you have Python 3.x installed.
   - For nnU-Net, follow the additional installation instructions provided in its documentation.

2. **Data Preparation:**  
   - Organize your microscopy images as outlined in the documentation.
   - Adjust any file paths in the scripts if necessary.

3. **Running the Code:**  
   - For whole image classification, run the corresponding training and evaluation scripts.
   - For cell detection and classification, execute the preprocessing script followed by the classification routine.
   - For cell segmentation, use the nnU-Net scripts provided to train and segment the images.

Dependencies
------------
- Python 3.x
- PyTorch
- TensorFlow (required for nnU-Net)
- OpenCV
- scikit-image
- Additional packages as specified in `requirements.txt`

Results and Further Analysis
------------------------------
The results and analyses derived from these experiments are detailed in the paper. This repository is designed to facilitate the replication of our experiments and to serve as a basis for further research in this area.

Contributing
------------
Contributions to this project are welcome. If you find issues or have suggestions for improvements, please open an issue or submit a pull request.

Contact
-------
For further questions regarding the code or the paper, please contact:
Maysam Orouskhani (morouskh@fredhutch.org) and Neelendu Dey (ndey@fredhutch.org)

Citation
--------
If you use this repository for your research, please cite our paper:
"Deep Learning Imaging Analysis to Identify Bacterial Metabolic States Associated with Carcinogen Production"

Thank you for your interest in our work!
