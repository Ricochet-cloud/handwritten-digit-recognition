#!/usr/bin/env python
"""
Handwritten Digit Recognition with Debug Mode
Shows preprocessed image and detailed analysis
"""

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

class DebugDigitRecognitionApp:
    """Digit recognition with visualization of preprocessing"""

    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition - Debug Mode")
        self.root.geometry("900x800")
        self.root.configure(bg='#f0f0f0')

        # Load model
        if not os.path.exists('digit_model.h5'):
            messagebox.showerror("Error", "Model file not found!")
            return

        self.model = keras.models.load_model('digit_model.h5')

        # Main frame
        main_frame = tk.Frame(root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = tk.Label(main_frame, text="Digit Recognition - Debug Mode",
                              font=("Arial", 14, "bold"), bg='#f0f0f0')
        title_label.pack(pady=5)

        # Canvas frame
        canvas_frame = tk.Frame(main_frame, bg='#f0f0f0')
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Draw canvas (left)
        left_frame = tk.Frame(canvas_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        canvas_label = tk.Label(left_frame, text="Draw Here (Original 400x400)",
                               font=("Arial", 10, "bold"), bg='#f0f0f0')
        canvas_label.pack(pady=3)

        self.canvas = tk.Canvas(left_frame, width=350, height=350,
                               bg='white', cursor="crosshair", relief=tk.SUNKEN, bd=2)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)

        self.image = Image.new("RGB", (350, 350), color=(255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None

        # Preview frame (right)
        right_frame = tk.Frame(canvas_frame, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        preview_label = tk.Label(right_frame, text="Processed (28x28 - Model Input)",
                                font=("Arial", 10, "bold"), bg='#f0f0f0')
        preview_label.pack(pady=3)

        # Preview canvas
        self.preview_canvas = tk.Canvas(right_frame, width=280, height=280,
                                       bg='lightgray', relief=tk.SUNKEN, bd=2)
        self.preview_canvas.pack()

        # Button frame
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, pady=10)

        recognize_btn = tk.Button(button_frame, text="Recognize",
                                 command=self.recognize_digit,
                                 font=("Arial", 11, "bold"),
                                 bg='#4CAF50', fg='white', padx=15, pady=8)
        recognize_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = tk.Button(button_frame, text="Clear",
                             command=self.clear_canvas,
                             font=("Arial", 11, "bold"),
                             bg='#FF9800', fg='white', padx=15, pady=8)
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Result frame
        result_frame = tk.Frame(main_frame, bg='white', relief=tk.SUNKEN, bd=2)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.result_label = tk.Label(result_frame, text="Draw to predict",
                                    font=("Arial", 16, "bold"), bg='white', fg='#333')
        self.result_label.pack(pady=10)

        self.confidence_label = tk.Label(result_frame, text="",
                                        font=("Arial", 11), bg='white', fg='#666')
        self.confidence_label.pack(pady=5)

        self.details_label = tk.Label(result_frame, text="",
                                     font=("Arial", 9), bg='white', fg='#666',
                                     justify=tk.LEFT)
        self.details_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    def on_canvas_press(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def on_canvas_drag(self, event):
        if self.last_x is not None and self.last_y is not None:
            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                   fill='black', width=12, capstyle=tk.ROUND)

            # Draw on PIL
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                          fill=(0, 0, 0), width=12)

        self.last_x = event.x
        self.last_y = event.y

    def preprocess_and_visualize(self, image):
        """
        Advanced preprocessing with step-by-step explanation
        """
        # Step 1: Convert to grayscale
        image_gray = image.convert('L')
        image_array = np.array(image_gray)

        # Step 2: Invert (white bg -> black bg for easier processing)
        image_array = 255 - image_array

        # Step 3: Apply contrast enhancement
        image_pil = Image.fromarray(image_array)
        enhancer = ImageEnhance.Contrast(image_pil)
        image_enhanced = enhancer.enhance(2.0)
        image_array = np.array(image_enhanced)

        # Step 4: Resize to 28x28
        image_resized = Image.fromarray(image_array).resize((28, 28), Image.Resampling.LANCZOS)
        image_array_28 = np.array(image_resized)

        # Step 5: Normalize
        image_normalized = image_array_28.astype("float32") / 255.0

        # Step 6: Apply threshold for clarity
        threshold = 0.2
        image_binary = np.where(image_normalized > threshold, image_normalized, 0)

        return image_binary, image_array_28

    def visualize_preprocessed(self, processed_image):
        """Show preprocessed image on preview canvas"""
        # Scale up for visualization (28x28 -> 280x280)
        scale = 10
        small = (processed_image * 255).astype(np.uint8)
        img_pil = Image.fromarray(small)
        img_scaled = img_pil.resize((280, 280), Image.Resampling.NEAREST)

        # Convert to PhotoImage
        from PIL import ImageTk
        photo = ImageTk.PhotoImage(img_scaled)

        self.preview_canvas.create_image(140, 140, image=photo)
        self.preview_canvas.image = photo

    def recognize_digit(self):
        """Recognize with detailed feedback"""
        try:
            # Preprocess
            processed, original_28 = self.preprocess_and_visualize(self.image)

            # Visualize
            self.visualize_preprocessed(processed)

            # Predict
            image_flat = processed.reshape(1, -1)
            prediction = self.model.predict(image_flat, verbose=0)
            predicted_digit = np.argmax(prediction[0])
            confidence = prediction[0][predicted_digit] * 100

            # Check if confidence is too low
            if confidence < 50:
                self.result_label.config(
                    text=f"Predicted: {predicted_digit} (Low confidence)",
                    fg='#D32F2F'
                )
            elif confidence < 70:
                self.result_label.config(
                    text=f"Predicted: {predicted_digit} (Medium confidence)",
                    fg='#F57C00'
                )
            else:
                self.result_label.config(
                    text=f"Predicted: {predicted_digit}",
                    fg='#2E7D32'
                )

            # Bar
            bar_length = int(confidence / 10)
            bar = "[" + "=" * bar_length + "-" * (10 - bar_length) + "]"
            self.confidence_label.config(
                text=f"Confidence: {confidence:.2f}%  {bar}"
            )

            # Top predictions
            top_5_idx = np.argsort(prediction[0])[-5:][::-1]
            details = "Top 5 predictions:\n"
            for rank, idx in enumerate(top_5_idx, 1):
                prob = prediction[0][idx] * 100
                mark = " <-- Predicted" if idx == predicted_digit else ""
                details += f"  {rank}. Digit {idx}: {prob:5.2f}%{mark}\n"

            self.details_label.config(text=details.strip())

            # Tips if low confidence
            if confidence < 70:
                tip = "\n\nTIP: Try drawing larger and more centered"
                self.details_label.config(text=details.strip() + tip, fg='#D32F2F')

        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")

    def clear_canvas(self):
        """Clear everything"""
        self.canvas.delete("all")
        self.preview_canvas.delete("all")
        self.image = Image.new("RGB", (350, 350), color=(255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.result_label.config(text="Draw to predict", fg='#333')
        self.confidence_label.config(text="")
        self.details_label.config(text="")

def main():
    root = tk.Tk()
    app = DebugDigitRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    print("="*60)
    print("DIGIT RECOGNITION - DEBUG MODE")
    print("="*60)
    print("\nThis version shows:")
    print("  - Original drawing (left)")
    print("  - Processed 28x28 image used by model (right)")
    print("  - Detailed preprocessing steps")
    print("  - Top 5 predictions")
    print("="*60 + "\n")
    main()
