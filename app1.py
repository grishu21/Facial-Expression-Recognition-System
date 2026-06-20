import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model(r'C:\Users\HP\Desktop\face_recognition_project\face_expression_model.h5')

# Define the emotion labels (adjust if your model uses different labels)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

predictions = [[9.5797648e-19, 1.2646153e-15, 4.2120511e-16, 1.0000000e+00, 
                6.9477077e-11, 3.3681499e-14, 6.1801882e-14]]

# Get the index of the highest probability
predicted_index = np.argmax(predictions)

# Map the index to the emotion label
predicted_emotion = emotion_labels[predicted_index]

# Load and preprocess the test image
image = Image.open(r'C:\Users\HP\Desktop\face_recognition_project\archive\test\happy\im0.png').convert('L')
image = image.resize((48, 48))
image_array = np.array(image).reshape(1, 48, 48, 1) / 255.0

# Get predictions
predictions = model.predict(image_array)

# Find the index of the highest probability
predicted_index = np.argmax(predictions)

# Get the emotion label
predicted_emotion = emotion_labels[predicted_index]

print(f"Predicted Emotion: {predicted_emotion}")

