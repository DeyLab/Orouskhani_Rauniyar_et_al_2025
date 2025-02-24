Classification of Microscopy Images using Deep Learning
========================================================

Overview
--------
This module contains code for classifying microscopy images of *Clostridium scindens* using deep learning models.
We employ three different CNN architectures:
1. **Pre-trained DenseNet121**
2. **Pre-trained ResNet50**
3. **CNN model from Scratch**

These models are trained to differentiate between different metabolic states of *C. scindens* in microscopy images.

Dataset
--------
The dataset consists of microscopy images divided into two main classes:
- **CS** (C. scindens in media alone)
- **CS_with_Cholic_acid** (C. scindens in media with cholic acid)

The dataset is split into:
- Training set (`training/`)
- Testing set (`testing/`)

All images are resized to **224x224 pixels** before feeding into the models.

Requirements
------------
Install the necessary dependencies using: pip install -r requirements.txt



