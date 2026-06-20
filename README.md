# Facial Expression Recognition System

## Overview

This project is a Deep Learning based Facial Expression Recognition System that detects human facial expressions from images using Convolutional Neural Networks (CNN). The system is built using TensorFlow/Keras and deployed through a Flask web application.

## Features

* Facial Expression Detection
* Deep Learning based CNN Model
* Image Upload Interface
* Real-time Prediction
* Flask Web Application
* User-Friendly Interface
* Model Evaluation and Testing

## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* Flask
* HTML
* CSS
* NumPy
* Pandas

## Project Structure

```text
face_recognition_project/
│
├── app.py
├── train_model.py
├── test_model.py
├── predict_and_evaluate.py
├── preprocess_dataset.py
├── requirements.txt
├── templates/
├── static/
├── uploads/
└── README.md
```

## Dataset

The model is trained on facial expression datasets containing various emotions such as:

* Happy
* Sad
* Angry
* Surprise
* Fear
* Neutral
* Disgust

## Installation

1. Clone the repository

```bash
git clone https://github.com/grishu21/face-recognition-web.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the application

```bash
python app.py
```

4. Open the browser and access:

```text
http://127.0.0.1:5000
```

## Model Performance

The CNN model is trained using image preprocessing, feature extraction, and classification techniques to identify facial expressions accurately.

## Future Enhancements

* Real-time Webcam Detection
* Improved CNN Architectures
* Mobile Application Integration
* Cloud Deployment
* Multi-face Recognition


