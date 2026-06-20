import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# Load the trained model
model = tf.keras.models.load_model('face_recognition_model.keras')
print("Model loaded successfully.")

# Define class labels
class_labels = {
    0: 'Angry',
    1: 'Disgusted',
    2: 'Fearful',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprised'
}
print("Class labels defined:", class_labels)

# Load and preprocess the image for prediction
image_path = r'C:\Users\HP\Desktop\face_recognition_project\archive\test\surprised\im87.png'  # Replace with your image path
img = load_img(image_path, target_size=(48, 48))  # Resize to match model input
img_array = img_to_array(img)
img_array = img_array / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make predictions
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# Map the predicted class to its label
predicted_label = class_labels[predicted_class[0]]
print(f"Predicted class: {predicted_label}")
