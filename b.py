import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('face_recognition_model.keras')  # Adjust path if necessary

# Load an image from file
img_path = r'C:\Users\HP\Desktop\face_recognition_project\uploads\surprised.jpg'  # Replace with the path to your test image
img = image.load_img(img_path, target_size=(48, 48))  # Adjust target size based on model input

# Preprocess the image to match the input shape of the model
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize image as done in training

# Predict the class of the image
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# Load class indices (if training data generator was used)
class_labels = train_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}  # Reverse the dictionary

# Get the predicted class name
predicted_class_name = class_labels[predicted_class[0]]

# Print the predicted class and name
print(f"Predicted class index: {predicted_class[0]}")
print(f"Predicted class name: {predicted_class_name}")
