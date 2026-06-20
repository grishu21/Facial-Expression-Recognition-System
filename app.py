from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the model once when the app starts to avoid loading it on every request
model = tf.keras.models.load_model(r'C:\Users\HP\Desktop\face_recognition_project\face_expression_model.h5')

# Emotions mapping (same as model output)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the file part exists
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Open and process the image
        image = Image.open(filepath).convert('L')  # Convert to grayscale
        image = image.resize((48, 48))  # Resize image to 48x48
        image_array = np.array(image).reshape(1, 48, 48, 1) / 255.0  # Normalize and reshape

        print(f"Image shape before prediction: {image_array.shape}")

        # Make predictions
        predictions = model.predict(image_array)
        print(f"Prediction Output: {predictions}")

        # Get the predicted class (highest probability)
        predicted_class = np.argmax(predictions, axis=-1)[0]
        predicted_emotion = emotions[predicted_class]
        confidence_scores = predictions.tolist()[0]  # Convert to list for response

        return jsonify({
            "predicted_emotion": predicted_emotion,
            "confidence_scores": confidence_scores
        })

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True,port=5500)
