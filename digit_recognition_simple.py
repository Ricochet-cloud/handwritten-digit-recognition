#!/usr/bin/env python
"""
Simple Handwritten Digit Recognition using MNIST dataset
This script trains a neural network to recognize handwritten digits
"""

try:
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import warnings
    warnings.filterwarnings('ignore')

    print("="*60)
    print("HANDWRITTEN DIGIT RECOGNITION SYSTEM")
    print("="*60)

    # Load MNIST dataset
    print("\n[1/4] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize and flatten
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    print(f"   Training samples: {len(x_train)}")
    print(f"   Test samples: {len(x_test)}")

    # Build model
    print("\n[2/4] Building neural network model...")
    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("   Model ready for training")

    # Train model
    print("\n[3/4] Training the model (10 epochs)...")
    history = model.fit(x_train, y_train, batch_size=128, epochs=10,
                       validation_split=0.1, verbose=0)
    print(f"   Final training accuracy: {history.history['accuracy'][-1]*100:.2f}%")

    # Evaluate
    print("\n[4/4] Evaluating on test dataset...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"   Test Accuracy: {test_accuracy*100:.2f}%")

    # Test with sample images
    print("\n" + "="*60)
    print("PREDICTION SAMPLES")
    print("="*60)

    for i in range(5):
        idx = np.random.randint(0, len(x_test))
        test_image = x_test[idx]
        true_digit = y_test[idx]

        # Predict
        prediction = model.predict(test_image.reshape(1, -1), verbose=0)
        pred_digit = np.argmax(prediction[0])
        confidence = prediction[0][pred_digit] * 100
        status = "CORRECT" if true_digit == pred_digit else "WRONG"

        print(f"\nSample {i+1}: {status}")
        print(f"  True digit: {true_digit}")
        print(f"  Predicted: {pred_digit} (Confidence: {confidence:.1f}%)")

    # Save model
    model.save('digit_model.h5')
    print("\n" + "="*60)
    print("Model saved successfully as 'digit_model.h5'")
    print("="*60)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
