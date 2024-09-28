Dataset link -"https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset" you can download for the making of model if use direct than download and use run gui file 

Car Colour Detection
Project Overview
The Car Colour Detection project aims to develop a machine learning model capable of identifying the color of cars in traffic images and counting the number of cars at a traffic signal. The model highlights blue cars with red rectangles and other colored cars with blue rectangles. Additionally, the model can detect the presence of people at the traffic signal and display the count of people present.

Features
-Car Color Detection : Identifies the color of cars in traffic images.
-Car Counting : Counts the number of cars at a traffic signal.
-People Detection : Detects and counts the number of people present at the traffic signal.
-Bounding Boxes : Draws red rectangles around blue cars and blue rectangles around other colored cars.
-GUI : Provides a graphical user interface for image and video input preview.

Technologies Used
-TensorFlow Keras : For building and training the car color detection model.
-OpenCV : For image processing and object detection.
-Tkinter : For creating the graphical user interface.
-MobileNet SSD : Pre-trained model for object detection.

## Installation
To get started with this project, clone the repository and install the necessary dependencies.


├── data
│   ├── raw
│   ├── processed
├── models
│   ├── COLOR_detection_model.keras
│   ├── deploy.prototxt
│   ├── mobilenet_iter_73000.caffemodel
├── src
│   ├── data_preprocessing.py
│   ├── gui.py
│   ├── model_training.py
│   ├── object_detection.py
├── README.md
├── requirements.txt


Please replace placeholders like `Bhawani Jangid` with your actual GitHub [username](https://github.com/Bhawani-jangid/Car-Colour-Detection/). If you need any additional sections or further customization, feel free to ask!

