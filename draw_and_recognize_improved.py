#!/usr/bin/env python
"""
Improved Handwritten Digit Recognition with Better Image Processing
"""

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

class ImprovedDigitRecognitionApp:
    """Improved digit recognition with better preprocessing"""

    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition - Improved")
        self.root.geometry("700x750")
        self.root.configure(bg='#f0f0f0')

        # Load trained model
        if not os.path.exists('digit_model.h5'):
            messagebox.showerror("Error", "Model file 'digit_model.h5' not found!")
            return

        self.model = keras.models.load_model('digit_model.h5')

        # Create main frame
        main_frame = tk.Frame(root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = tk.Label(main_frame, text="Draw a Digit (0-9) - Improved Version",
                              font=("Arial", 14, "bold"), bg='#f0f0f0')
        title_label.pack(pady=5)

        # Instructions
        instr_label = tk.Label(main_frame,
                              text="Draw large, centered digits similar to MNIST style",
                              font=("Arial", 10), bg='#f0f0f0', fg='#666')
        instr_label.pack(pady=5)

        # Canvas for drawing
        self.canvas = tk.Canvas(main_frame, width=400, height=400,
                               bg='white', cursor="crosshair", relief=tk.SUNKEN, bd=2)
        self.canvas.pack(padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)

        # Create PIL image
        self.image = Image.new("RGB", (400, 400), color=(255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.brush_size = 15

        # Button frame
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, pady=10)

        recognize_btn = tk.Button(button_frame, text="Recognize",
                                 command=self.recognize_digit,
                                 font=("Arial", 11, "bold"),
                                 bg='#4CAF50', fg='white',
                                 padx=15, pady=8)
        recognize_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = tk.Button(button_frame, text="Clear",
                             command=self.clear_canvas,
                             font=("Arial", 11, "bold"),
                             bg='#FF9800', fg='white',
                             padx=15, pady=8)
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Brush size
        size_label = tk.Label(button_frame, text="Brush size:", bg='#f0f0f0', font=("Arial", 10))
        size_label.pack(side=tk.LEFT, padx=(20, 5))

        self.size_var = tk.IntVar(value=15)
        size_spinbox = tk.Spinbox(button_frame, from_=5, to=25, width=3,
                                 textvariable=self.size_var, font=("Arial", 10))
        size_spinbox.pack(side=tk.LEFT)

        # Result frame
        result_frame = tk.Frame(main_frame, bg='white', relief=tk.SUNKEN, bd=2)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Prediction result
        self.result_label = tk.Label(result_frame, text="Draw a digit to predict",
                                    font=("Arial", 14, "bold"), bg='white', fg='#333')
        self.result_label.pack(pady=15)

        # Confidence
        self.confidence_label = tk.Label(result_frame, text="",
                                        font=("Arial", 11), bg='white', fg='#666')
        self.confidence_label.pack(pady=5)

        # Probability details
        self.details_label = tk.Label(result_frame, text="",
                                     font=("Arial", 9), bg='white', fg='#666',
                                     justify=tk.LEFT)
        self.details_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    def on_canvas_press(self, event):
        """Handle canvas mouse press"""
        self.last_x = event.x
        self.last_y = event.y

    def on_canvas_drag(self, event):
        """Handle canvas mouse drag - draw with smooth brush"""
        if self.last_x is not None and self.last_y is not None:
            brush_size = self.size_var.get()

            # Draw on tkinter canvas
            self.canvas.create_oval(event.x - brush_size//2, event.y - brush_size//2,
                                   event.x + brush_size//2, event.y + brush_size//2,
                                   fill='black', outline='black')

            # Draw on PIL image
            self.draw.ellipse([event.x - brush_size//2, event.y - brush_size//2,
                              event.x + brush_size//2, event.y + brush_size//2],
                             fill=(0, 0, 0))

        self.last_x = event.x
        self.last_y = event.y

    def preprocess_image(self, image):
        """
        Preprocess drawn image to match MNIST format
        """
        # Convert to grayscale
        image_gray = image.convert('L')

        # Apply slight blur for smoothing
        image_gray = image_gray.filter(ImageFilter.GaussianBlur(radius=1))

        # Resize to 28x28
        image_small = image_gray.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to numpy array
        image_array = np.array(image_small).astype("float32")

        # Invert colors (white background, black digit)
        image_array = 255 - image_array

        # Normalize to [0, 1]
        image_array = image_array / 255.0

        # Apply threshold to make it more binary
        threshold = 0.3
        image_array = np.where(image_array > threshold, image_array, 0)

        return image_array

    def recognize_digit(self):
        """Recognize the drawn digit with improved preprocessing"""
        try:
            # Preprocess image
            processed_image = self.preprocess_image(self.image)

            # Flatten
            image_flat = processed_image.reshape(1, -1)

            # Make prediction
            prediction = self.model.predict(image_flat, verbose=0)
            predicted_digit = np.argmax(prediction[0])
            confidence = prediction[0][predicted_digit] * 100

            # Update result display
            self.result_label.config(
                text=f"Predicted: {predicted_digit}",
                fg='#2E7D32',
                font=("Arial", 18, "bold")
            )

            # Confidence bar
            bar_length = int(confidence / 10)
            bar = "[" + "=" * bar_length + "-" * (10 - bar_length) + "]"
            self.confidence_label.config(
                text=f"Confidence: {confidence:.2f}%  {bar}",
                fg='#1976D2' if confidence > 80 else '#F57C00'
            )

            # Show top 3 predictions
            top_3_idx = np.argsort(prediction[0])[-3:][::-1]
            details_text = "Top 3 predictions:\n\n"
            for rank, idx in enumerate(top_3_idx, 1):
                prob_percent = prediction[0][idx] * 100
                bar_len = int(prob_percent / 5)
                prob_bar = "[" + "=" * bar_len + "-" * (20 - bar_len) + "]"
                details_text += f"  {rank}. Digit {idx}: {prob_percent:5.2f}%  {prob_bar}\n"

            self.details_label.config(text=details_text.strip())

        except Exception as e:
            messagebox.showerror("Error", f"Recognition failed: {str(e)}")

    def clear_canvas(self):
        """Clear the canvas"""
        self.canvas.delete("all")
        self.image = Image.new("RGB", (400, 400), color=(255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.result_label.config(text="Draw a digit to predict", fg='#333')
        self.confidence_label.config(text="")
        self.details_label.config(text="")

def main():
    root = tk.Tk()
    app = ImprovedDigitRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    print("="*60)
    print("IMPROVED HANDWRITTEN DIGIT RECOGNITION")
    print("="*60)
    print("\nTips for better recognition:")
    print("  - Draw digits large and centered on canvas")
    print("  - Use similar style to how you write normally")
    print("  - Adjust brush size if needed (5-25 pixels)")
    print("  - Draw complete digits without gaps")
    print("="*60 + "\n")
    main()
