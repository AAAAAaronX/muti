Overview

This project implements an improved multi-modal fusion network for predicting mechanical properties of fiber composite materials. It combines image data, structural features (fiber shapes via one-hot encoding), and material properties to predict properties like E1, E2, v23, etc. The model uses PyTorch and incorporates residual blocks, cross-modal attention, uncertainty estimation, and Grad-CAM for visualization.
The code is modularized into several Python files for better organization and maintainability. It supports training, evaluation, visualization, and single-sample prediction.

Requirements

Python 3.8+
PyTorch
Torchvision
NumPy
Pandas
Scikit-learn
Matplotlib
OpenCV (cv2)
Pillow (PIL)