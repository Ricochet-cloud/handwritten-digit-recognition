#!/usr/bin/env python
"""
Interactive Canvas - Draw Digits and Get AI Recognition
Uses tkinter for GUI and trained neural network for recognition
"""

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

class DigitRecognitionApp:
    """Main application class for handwritten digit recognition"""

    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("600x700")
        self.root.configure(bg='#f0f0f0')

        # Load trained model
        if os.path.exists('digit_model.h5'):
            self.model = keras.models.load_model('digit_model.h5')
        else:
            messagebox.showerror("Error", "Model file not found!")
            return

        # Create main frame
        main_frame = tk.Frame(root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = tk.Label(main_frame, text="Draw a Digit (0-9)",
                              font=("Arial", 16, "bold"), bg='#f0f0f0')
        title_label.pack(pady=10)

        # Canvas for drawing
        self.canvas = tk.Canvas(main_frame, width=400, height=400,
                               bg='white', cursor="crosshair")
        self.canvas.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)

        # Create PIL image for processing
        self.image = Image.new("L", (400, 400), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None

        # Button frame
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, pady=10)

        # Recognize button
        recognize_btn = tk.Button(button_frame, text="Recognize Digit",
                                 command=self.recognize_digit,
                                 font=("Arial", 12, "bold"),
                                 bg='#4CAF50', fg='white',
                                 padx=20, pady=10)
        recognize_btn.pack(side=tk.LEFT, padx=5)

        # Clear button
        clear_btn = tk.Button(button_frame, text="Clear Canvas",
                             command=self.clear_canvas,
                             font=("Arial", 12, "bold"),
                             bg='#FF9800', fg='white',
                             padx=20, pady=10)
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Result frame
        result_frame = tk.Frame(main_frame, bg='white', relief=tk.SUNKEN, bd=2)
        result_frame.pack(fill=tk.X, pady=10)

        # Prediction result
        self.result_label = tk.Label(result_frame, text="Draw something to predict",
                                    font=("Arial", 12), bg='white', fg='#666')
        self.result_label.pack(pady=10)

        # Confidence bar
        self.confidence_label = tk.Label(result_frame, text="",
                                        font=("Arial", 10), bg='white', fg='#999')
        self.confidence_label.pack(pady=5)

        # Probability details
        self.details_label = tk.Label(result_frame, text="",
                                     font=("Arial", 9), bg='white', fg='#999',
                                     justify=tk.LEFT)
        self.details_label.pack(pady=5, padx=10)

    def on_canvas_press(self, event):
        """Handle canvas mouse press event"""
        self.last_x = event.x
        self.last_y = event.y

    def on_canvas_drag(self, event):
        """Handle canvas mouse drag event - draw line"""
        if self.last_x is not None and self.last_y is not None:
            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                   fill='black', width=8, capstyle=tk.ROUND, smooth=True)

            # Draw on PIL image
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                          fill=0, width=8)

        self.last_x = event.x
        self.last_y = event.y

    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete("all")
        self.image = Image.new("L", (400, 400), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.result_label.config(text="Canvas cleared. Draw something!")
        self.confidence_label.config(text="")
        self.details_label.config(text="")

    def recognize_digit(self):
        """Recognize the digit drawn on canvas"""
        # Resize image to 28x28
        small_image = self.image.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to numpy array
        image_array = np.array(small_image).astype("float32") / 255.0

        # Invert (white to black)
        image_array = 1.0 - image_array

        # Flatten
        image_flat = image_array.reshape(1, -1)

        # Make prediction
        prediction = self.model.predict(image_flat, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = prediction[0][predicted_digit] * 100

        # Update result display
        self.result_label.config(
            text=f"Predicted Digit: {predicted_digit}",
            font=("Arial", 18, "bold"),
            fg='#2E7D32'
        )

        # Confidence bar
        bar_length = int(confidence / 10)
        bar = "[" + "=" * bar_length + "-" * (10 - bar_length) + "]"
        self.confidence_label.config(
            text=f"Confidence: {confidence:.2f}% {bar}"
        )

        # Show top 3 predictions
        top_3_idx = np.argsort(prediction[0])[-3:][::-1]
        details_text = "Top 3 predictions:\n"
        for rank, idx in enumerate(top_3_idx, 1):
            prob_percent = prediction[0][idx] * 100
            details_text += f"  {rank}. Digit {idx}: {prob_percent:.2f}%\n"

        self.details_label.config(text=details_text.strip())

def main():
    """Main function"""
    root = tk.Tk()
    app = DigitRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    print("="*60)
    print("HANDWRITTEN DIGIT RECOGNITION - INTERACTIVE CANVAS")
    print("="*60)
    print("\nStarting GUI application...")
    print("Instructions:")
    print("  1. Draw a digit (0-9) on the canvas")
    print("  2. Click 'Recognize Digit' button")
    print("  3. The AI will predict your digit")
    print("  4. Click 'Clear Canvas' to draw again")
    print("\nNote: The model was trained on 60,000 handwritten digits")
    print("      and achieves 97.9% accuracy on test data.")
    print("="*60 + "\n")

    main()
