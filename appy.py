from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import io

app = Flask(__name__)

# Load the model (ensure it's loaded before any predictions are made)
model = load_model('face_expression_model.h5')

@app.route('/')
def home():
    return "Welcome to the Face Expression Recognition API"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read and preprocess the image
        img = load_img(io.BytesIO(file.read()), color_mode='grayscale', target_size=(48, 48))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions using the loaded model
        predictions = model.predict(img_array)

        # Convert predictions into a human-readable format
        predicted_class = np.argmax(predictions, axis=1)
        class_names = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        prediction = class_names[predicted_class[0]]

        # Return the prediction
        return jsonify({
            'prediction': prediction,
            'confidence': float(np.max(predictions))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
