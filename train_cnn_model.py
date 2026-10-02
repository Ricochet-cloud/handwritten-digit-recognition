#!/usr/bin/env python
"""
Train a powerful CNN model for digit recognition
CNN (Convolutional Neural Network) achieves 99%+ accuracy
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TRAINING CNN MODEL FOR DIGIT RECOGNITION")
print("="*70)

# Load MNIST dataset
print("\n[1/5] Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Reshape to (num_samples, 28, 28, 1) for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(f"   Training samples: {len(x_train)}")
print(f"   Test samples: {len(x_test)}")
print(f"   Input shape: {x_train.shape[1:]}")

# Build CNN model
print("\n[2/5] Building CNN model...")
model = keras.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Third convolutional block
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.Dropout(0.25),

    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("   Model architecture:")
model.summary()

# Train model
print("\n[3/5] Training model (15 epochs)...")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=15,
    validation_split=0.1,
    verbose=1
)

# Evaluate
print("\n[4/5] Evaluating on test dataset...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test Accuracy: {test_accuracy*100:.2f}%")

# Test with sample digits
print("\n[5/5] Testing with sample digits...")
sample_accuracy = 0
for digit in range(10):
    idx = np.where(y_test == digit)[0][0]
    test_image = x_test[idx:idx+1]

    prediction = model.predict(test_image, verbose=0)
    predicted_digit = np.argmax(prediction[0])
    confidence = prediction[0][predicted_digit] * 100

    is_correct = "YES" if predicted_digit == digit else "NO"
    print(f"   Digit {digit}: Predicted {predicted_digit} ({confidence:.1f}%) - {is_correct}")

    if predicted_digit == digit:
        sample_accuracy += 1

print(f"   Sample Accuracy: {sample_accuracy*10:.0f}%")

# Save model
print("\n" + "="*70)
print("Saving CNN model as 'digit_cnn_model.h5'...")
model.save('digit_cnn_model.h5')
print("="*70)

print("\nCNN Model Performance Summary:")
print(f"  - Architecture: 3 Conv layers + 2 Dense layers")
print(f"  - Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  - Model Type: CNN (Convolutional Neural Network)")
print(f"  - Expected Accuracy: 99%+")
print(f"\nNOTE: Use 'digit_cnn_model.h5' for much better accuracy!")
print("="*70 + "\n")
