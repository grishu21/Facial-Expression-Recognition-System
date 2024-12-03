import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Load the saved model
model = load_model('face_expression_model.h5')

# Map labels back to expressions
expression_mapping = {
    0: 'angry',
    1: 'disgusted',
    2: 'fearful',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprised'
}

# Path to test image (ensure the path is correct)
img_path = r'C:\Users\HP\Desktop\face_recognition_project\archive\test\happy\im2.png'  # Replace with your image path

# Preprocess the image (resize, convert to grayscale, normalize)
img = load_img(img_path, target_size=(48, 48), color_mode='grayscale')  # Resize to 48x48 pixels and grayscale
img_array = img_to_array(img) / 255.0  # Convert to array and normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict the expression
predictions = model.predict(img_array)  # Get model predictions
predicted_class = np.argmax(predictions)  # Get the index of the highest probability
predicted_expression = expression_mapping[predicted_class]  # Map index to expression

# Print the result of the individual image prediction
print(f"Predicted Expression: {predicted_expression}")

# --- Evaluation Part: Evaluate the model on the test dataset ---

# Path to the test dataset
TEST_DATASET_DIR = 'C:/Users/HP/Desktop/face_recognition_project/archive/test'

# Parameters (same as training data)
IMG_SIZE = 48
NUM_CLASSES = 7

# Initialize test data and labels lists
test_data = []
test_labels = []

# Load and preprocess test images
for label in expression_mapping.keys():  # Corrected to iterate over the keys (which are integers)
    expression_name = expression_mapping[label]  # Get the corresponding expression name (string)
    expression_folder = os.path.join(TEST_DATASET_DIR, expression_name)  # Construct the folder path
    for img_name in os.listdir(expression_folder):
        img_path = os.path.join(expression_folder, img_name)
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale')
        img_array = img_to_array(img) / 255.0  # Normalize
        test_data.append(img_array)
        test_labels.append(label)

# Convert to NumPy arrays
test_data = np.array(test_data, dtype="float32")
test_labels = np.array(test_labels, dtype="int")

# One-hot encode the test labels
test_labels = to_categorical(test_labels, num_classes=NUM_CLASSES)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_data, test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")
