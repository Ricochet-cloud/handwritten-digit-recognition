#!/usr/bin/env python
"""
Handwritten Digit Recognition with CNN Model (99.4% Accuracy)
Much more accurate than Dense neural network
"""

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

class CNNDigitRecognitionApp:
    """CNN-based digit recognition with 99.4% accuracy"""

    def __init__(self, root):
        self.root = root
        self.root.title("CNN Digit Recognition - 99.4% Accurate")
        self.root.geometry("950x850")
        self.root.configure(bg='#f0f0f0')

        # Load CNN model
        if not os.path.exists('digit_cnn_model.h5'):
            messagebox.showerror("Error", "CNN model not found!\nRun 'train_cnn_model.py' first!")
            return

        print("Loading CNN model...")
        self.model = keras.models.load_model('digit_cnn_model.h5')
        print("CNN model loaded successfully!")

        self.recognition_count = 0

        # Main frame
        main_frame = tk.Frame(root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = tk.Label(main_frame, text="CNN Digit Recognition (99.4% Accurate)",
                              font=("Arial", 16, "bold"), bg='#f0f0f0', fg='#1976D2')
        title_label.pack(pady=5)

        # Instructions
        instr = tk.Label(main_frame,
                        text="Draw digit clearly in center | CNN model achieves 99.4% accuracy on MNIST",
                        font=("Arial", 10), bg='#f0f0f0', fg='#666')
        instr.pack(pady=3)

        # Canvas area
        canvas_frame = tk.Frame(main_frame, bg='#f0f0f0')
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Draw canvas (left)
        left_frame = tk.Frame(canvas_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(left_frame, text="Draw (320x320)", font=("Arial", 11, "bold"), bg='#f0f0f0').pack(pady=3)
        self.canvas = tk.Canvas(left_frame, width=350, height=350,
                               bg='white', cursor="crosshair", relief=tk.SUNKEN, bd=2)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)

        self.image = Image.new("RGB", (320, 320), color=(255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None

        # Preview (right)
        right_frame = tk.Frame(canvas_frame, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(right_frame, text="Model Input (28x28)", font=("Arial", 11, "bold"), bg='#f0f0f0').pack(pady=3)
        self.preview_canvas = tk.Canvas(right_frame, width=280, height=280,
                                       bg='lightgray', relief=tk.SUNKEN, bd=2)
        self.preview_canvas.pack()

        # Buttons
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, pady=10)

        tk.Button(button_frame, text="Recognize",
                 command=self.recognize_digit,
                 font=("Arial", 12, "bold"),
                 bg='#4CAF50', fg='white', padx=20, pady=10).pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="Clear",
                 command=self.clear_canvas,
                 font=("Arial", 12, "bold"),
                 bg='#FF9800', fg='white', padx=20, pady=10).pack(side=tk.LEFT, padx=5)

        # Result frame
        result_frame = tk.Frame(main_frame, bg='white', relief=tk.SUNKEN, bd=2)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Prediction
        self.result_label = tk.Label(result_frame, text="Draw digit",
                                    font=("Arial", 32, "bold"), bg='white', fg='#333')
        self.result_label.pack(pady=15)

        # Confidence
        self.confidence_label = tk.Label(result_frame, text="",
                                        font=("Arial", 12), bg='white', fg='#1976D2')
        self.confidence_label.pack(pady=5)

        # Top predictions
        self.details_label = tk.Label(result_frame, text="",
                                     font=("Arial", 10), bg='white', fg='#666',
                                     justify=tk.LEFT)
        self.details_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.last_processed = None

    def on_canvas_press(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def on_canvas_drag(self, event):
        if self.last_x is not None and self.last_y is not None:
            # Scale coordinates (canvas is scaled)
            scale_x = 320 / 350
            scale_y = 320 / 350

            x1 = int(self.last_x * scale_x)
            y1 = int(self.last_y * scale_y)
            x2 = int(event.x * scale_x)
            y2 = int(event.y * scale_y)

            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                   fill='black', width=12, capstyle=tk.ROUND)

            # Draw on PIL (constrain to 320x320)
            if 0 <= x1 < 320 and 0 <= y1 < 320 and 0 <= x2 < 320 and 0 <= y2 < 320:
                self.draw.line([x1, y1, x2, y2], fill=(0, 0, 0), width=12)

        self.last_x = event.x
        self.last_y = event.y

    def preprocess_for_cnn(self, image):
        """
        Preprocess image for CNN model
        CNN expects 28x28 grayscale images normalized to [0, 1]
        """
        # Convert to grayscale
        img_gray = image.convert('L')
        img_array = np.array(img_gray, dtype=np.float32)

        # Invert (black digit on white background)
        img_array = 255 - img_array

        # Enhance contrast
        pil_img = Image.fromarray(img_array.astype(np.uint8))
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(2.0)
        img_array = np.array(pil_img, dtype=np.float32)

        # Resize to 28x28
        pil_img = Image.fromarray(img_array.astype(np.uint8))
        img_small = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
        img_array_28 = np.array(img_small, dtype=np.float32)

        # Normalize to [0, 1]
        img_normalized = img_array_28 / 255.0

        # Reshape for CNN: (28, 28, 1)
        img_cnn = img_normalized.reshape(28, 28, 1)

        return img_cnn, img_normalized

    def visualize_image(self, img_array, canvas):
        """Display 28x28 image on canvas"""
        img_vis = (img_array * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_vis)
        img_scaled = img_pil.resize((280, 280), Image.Resampling.NEAREST)

        from PIL import ImageTk
        photo = ImageTk.PhotoImage(img_scaled)

        canvas.create_image(140, 140, image=photo)
        canvas.image = photo

    def recognize_digit(self):
        """Recognize digit using CNN"""
        try:
            self.recognition_count += 1

            # Preprocess
            img_cnn, img_vis = self.preprocess_for_cnn(self.image)
            self.last_processed = img_vis

            # Visualize
            self.visualize_image(img_vis, self.preview_canvas)

            # Predict (CNN expects batch of images)
            prediction = self.model.predict(np.array([img_cnn]), verbose=0)
            predicted_digit = np.argmax(prediction[0])
            confidence = prediction[0][predicted_digit] * 100

            # Display result with color based on confidence
            if confidence >= 95:
                color = '#2E7D32'  # Dark green
                quality = "Very High"
            elif confidence >= 85:
                color = '#1976D2'  # Blue
                quality = "High"
            elif confidence >= 70:
                color = '#F57C00'  # Orange
                quality = "Medium"
            else:
                color = '#D32F2F'  # Red
                quality = "Low"

            self.result_label.config(text=f"{predicted_digit}", fg=color)

            # Confidence bar
            bar_len = int(confidence / 10)
            bar = "[" + "█" * bar_len + "░" * (10 - bar_len) + "]"
            self.confidence_label.config(
                text=f"Confidence: {confidence:.2f}% - {quality}  {bar}",
                fg=color
            )

            # Top 5 predictions
            top_5_idx = np.argsort(prediction[0])[-5:][::-1]
            details = "Top 5 Predictions:\n\n"
            for i, idx in enumerate(top_5_idx, 1):
                prob = prediction[0][idx] * 100
                bar_len2 = int(prob / 5)
                prob_bar = "[" + "█" * bar_len2 + "░" * (20 - bar_len2) + "]"
                mark = " ← CNN Selected" if idx == predicted_digit else ""
                details += f"  {i}. Digit [{idx}]: {prob:5.2f}%  {prob_bar}{mark}\n"

            self.details_label.config(text=details.strip())

        except Exception as e:
            messagebox.showerror("Error", f"Recognition failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def clear_canvas(self):
        """Clear everything"""
        self.canvas.delete("all")
        self.preview_canvas.delete("all")
        self.image = Image.new("RGB", (320, 320), color=(255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.result_label.config(text="Draw digit", fg='#333')
        self.confidence_label.config(text="")
        self.details_label.config(text="")

def main():
    root = tk.Tk()
    app = CNNDigitRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    print("="*70)
    print("CNN HANDWRITTEN DIGIT RECOGNITION")
    print("="*70)
    print("\nModel Information:")
    print("  - Type: Convolutional Neural Network (CNN)")
    print("  - Accuracy: 99.4% on MNIST test set")
    print("  - Architecture: 3 Conv layers + 2 Dense layers")
    print("  - Training Data: 60,000 handwritten digits")
    print("="*70 + "\n")
    main()
