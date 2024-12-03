import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

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

# Path to test image (ensure the path is correct)r"C:\Users\HP\Des
img_path = r'C:\Users\HP\Desktop\face_recognition_project\archive\test\happy\im2.png'  # Replace with your image path

# Preprocess the image (resize, convert to grayscale, normalize)
img = load_img(img_path, target_size=(48, 48), color_mode='grayscale')  # Resize to 48x48 pixels and grayscale
img_array = img_to_array(img) / 255.0  # Convert to array and normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict the expression
predictions = model.predict(img_array)  # Get model predictions
predicted_class = np.argmax(predictions)  # Get the index of the highest probability
predicted_expression = expression_mapping[predicted_class]  # Map index to expression

# Print the result
print(f"Predicted Expression: {predicted_expression}")
