# SimpleEmotionClassifier
A Simple Emotion Classification system using Facial Landmarks from the Dlib libary and Tensorflow to build and train a nural network.

Use
---
1. Download and install prerequisites.
2. Make a Image directory with indivual classification folders.
3. Move training images to their classification folders.
4. Train the classifier using ```python train 'Image Directory' 'Image File Type' 'Dlib Shape Predictor Path' 'Classifier Save Location' 'Number of Epochs'```
5. Get the classification of Image using ```python classify 'Image Path' 'Shape Predictor Path' 'Number of Classes' 'Session Save Path'```

Prerequisites
-------------
* Python 2.7
* OpenCV
* Dlib
* Numpy
* TensorFlow
* Dlib Shape Predictor file

TO-DO
-----
* Normalization on training input images
* Normalize testing image input
