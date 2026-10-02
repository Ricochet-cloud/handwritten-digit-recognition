import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load MNIST dataset - 28x28 grayscale images of handwritten digits
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1] range
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten 28x28 images to 784-dimensional vectors
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# Build neural network model
print("\nBuilding neural network model...")
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")  # 10 classes for digits 0-9
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
print("\nTraining the model...")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

# Evaluate on test dataset
print("\nEvaluating on test dataset...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Function to predict handwritten digits from images
def predict_digit(image):
    """
    Predict the digit in a given image.
    image: 28x28 numpy array (values 0-255)
    """
    # Normalize the image
    image = image.astype("float32") / 255.0
    # Flatten to 784 dimensions
    image_flat = image.reshape(1, -1)
    # Make prediction
    prediction = model.predict(image_flat, verbose=0)
    predicted_digit = np.argmax(prediction[0])
    confidence = prediction[0][predicted_digit] * 100
    return predicted_digit, confidence

# Test with some sample images from the test set
print("\n" + "="*50)
print("Testing with sample handwritten digits")
print("="*50)

for i in range(5):
    # Take a random test image
    idx = np.random.randint(0, len(x_test))
    test_image = (x_test[idx].reshape(28, 28) * 255).astype(np.uint8)
    true_label = y_test[idx]

    # Make prediction
    predicted_digit, confidence = predict_digit(test_image)

    print(f"\nSample {i+1}:")
    print(f"  True digit: {true_label}")
    print(f"  Predicted digit: {predicted_digit}")
    print(f"  Confidence: {confidence:.2f}%")
    print(f"  Result: {'✓ Correct' if true_label == predicted_digit else '✗ Wrong'}")

# Visualize some predictions
print("\n" + "="*50)
print("Visualizing predictions...")
print("="*50)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Handwritten Digit Recognition - Sample Predictions', fontsize=14, fontweight='bold')

for idx, ax in enumerate(axes.flat):
    # Get random test image
    i = np.random.randint(0, len(x_test))
    test_image = x_test[i].reshape(28, 28)
    true_label = y_test[i]

    # Make prediction
    predicted_digit, confidence = predict_digit((test_image * 255).astype(np.uint8))

    # Display image
    ax.imshow(test_image, cmap='gray')
    color = 'green' if true_label == predicted_digit else 'red'
    ax.set_title(f'True: {true_label}, Pred: {predicted_digit}\n({confidence:.1f}%)',
                 color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('digit_recognition_results.png', dpi=100, bbox_inches='tight')
print("\nVisualization saved as 'digit_recognition_results.png'")
plt.show()

# Save the trained model
model.save('digit_recognition_model.h5')
print("\nModel saved as 'digit_recognition_model.h5'")

print("\n" + "="*50)
print("Handwritten Digit Recognition - Complete!")
print("="*50)
