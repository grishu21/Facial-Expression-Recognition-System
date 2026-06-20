import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Define directories
train_dir = r'C:\Users\HP\Desktop\face_recognition_project\archive\train'
val_dir = r'C:\Users\HP\Desktop\face_recognition_project\archive\test'

# Create ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(48, 48), batch_size=32, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(48, 48), batch_size=32, class_mode='categorical')

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define ModelCheckpoint to save best model
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(train_generator, validation_data=val_generator, epochs=50, callbacks=[checkpoint])

# Save the model after training
model.save('face_recognition_model.keras')

# Evaluate on test set if available
test_loss, test_accuracy = model.evaluate(val_generator)  # Or use a separate test dataset
print(f"Test Accuracy: {test_accuracy}, Test Loss: {test_loss}")


