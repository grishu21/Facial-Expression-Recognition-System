import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import models, layers  # Correct imports
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define dataset directory
DATASET_DIR = 'C:\\Users\\HP\\Downloads\\archive'  # Replace with your dataset folder path

# Parameters
IMG_SIZE = 48  # FER-2013 images are 48x48
NUM_CLASSES = 7  # Number of expressions (Angry, Happy, Neutral, etc.)

# Initialize data and labels lists
data = []
labels = []

# Map expressions to numerical labels
expression_mapping = {
    'angry': 0,
    'disgusted': 1,
    'fearful': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprised': 6
}

# Load and preprocess images
print("Loading and preprocessing images...")
for expression, label in expression_mapping.items():
    expression_folder = os.path.join(DATASET_DIR, 'train', expression)
    for img_name in os.listdir(expression_folder):
        img_path = os.path.join(expression_folder, img_name)
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale')
        img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        data.append(img_array)
        labels.append(label)

# Convert to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels, dtype="int")

# One-hot encode labels
labels = to_categorical(labels, num_classes=NUM_CLASSES)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

print("Preprocessing completed.")
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# Model creation (use 'models' correctly)
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
model = create_model()
model.summary()  # To check the model architecture

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,  # Number of epochs to train
    batch_size=64,  # Batch size for training
    verbose=1
)

# Save the model
model.save('face_expression_model.h5')
print("Model training completed and saved as 'face_expression_model.h5'.")


# Load the saved model
from tensorflow.keras.models import load_model

model = load_model('face_expression_model.h5')

# Evaluate the model on validation data
loss, accuracy = model.evaluate(X_val, y_val, verbose=1)
print(f"Validation Accuracy: {accuracy:.4f}, Validation Loss: {loss:.4f}")

import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Trend')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Trend')

plt.show()

