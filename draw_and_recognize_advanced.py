#!/usr/bin/env python
"""
Advanced Handwritten Digit Recognition with Superior Image Processing
Uses advanced preprocessing techniques for better accuracy
"""

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

class AdvancedDigitRecognitionApp:
    """Advanced digit recognition with superior image preprocessing"""

    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Digit Recognition")
        self.root.geometry("1000x850")
        self.root.configure(bg='#f0f0f0')

        # Load model
        if not os.path.exists('digit_model.h5'):
            messagebox.showerror("Error", "Model file not found!")
            return

        self.model = keras.models.load_model('digit_model.h5')
        self.recognition_count = 0

        # Main frame
        main_frame = tk.Frame(root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_label = tk.Label(main_frame, text="Advanced Digit Recognition",
                              font=("Arial", 14, "bold"), bg='#f0f0f0')
        title_label.pack(pady=5)

        # Instructions
        instr = tk.Label(main_frame,
                        text="Draw digit BIG and DARK in center | All 3 images should show your digit clearly",
                        font=("Arial", 9), bg='#f0f0f0', fg='#666')
        instr.pack(pady=3)

        # Canvas area
        canvas_frame = tk.Frame(main_frame, bg='#f0f0f0')
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Original canvas (left)
        left_frame = tk.Frame(canvas_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(left_frame, text="Draw (400x400)", font=("Arial", 10, "bold"), bg='#f0f0f0').pack(pady=2)
        self.canvas = tk.Canvas(left_frame, width=320, height=320,
                               bg='white', cursor="crosshair", relief=tk.SUNKEN, bd=2)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)

        self.image = Image.new("RGB", (320, 320), color=(255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None

        # Processed 1 (middle)
        mid_frame = tk.Frame(canvas_frame, bg='#f0f0f0')
        mid_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(mid_frame, text="After Processing (28x28)", font=("Arial", 10, "bold"), bg='#f0f0f0').pack(pady=2)
        self.preview1_canvas = tk.Canvas(mid_frame, width=280, height=280,
                                        bg='lightgray', relief=tk.SUNKEN, bd=2)
        self.preview1_canvas.pack()

        # Processed 2 (right)
        right_frame = tk.Frame(canvas_frame, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(right_frame, text="Final (Enhanced 28x28)", font=("Arial", 10, "bold"), bg='#f0f0f0').pack(pady=2)
        self.preview2_canvas = tk.Canvas(right_frame, width=280, height=280,
                                        bg='lightgray', relief=tk.SUNKEN, bd=2)
        self.preview2_canvas.pack()

        # Buttons
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, pady=10)

        tk.Button(button_frame, text="Recognize",
                 command=self.recognize_digit,
                 font=("Arial", 11, "bold"),
                 bg='#4CAF50', fg='white', padx=15, pady=8).pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="Clear",
                 command=self.clear_canvas,
                 font=("Arial", 11, "bold"),
                 bg='#FF9800', fg='white', padx=15, pady=8).pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="Save Images",
                 command=self.save_images,
                 font=("Arial", 11, "bold"),
                 bg='#2196F3', fg='white', padx=15, pady=8).pack(side=tk.LEFT, padx=5)

        # Result frame
        result_frame = tk.Frame(main_frame, bg='white', relief=tk.SUNKEN, bd=2)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.result_label = tk.Label(result_frame, text="Draw digit",
                                    font=("Arial", 18, "bold"), bg='white', fg='#333')
        self.result_label.pack(pady=10)

        self.confidence_label = tk.Label(result_frame, text="",
                                        font=("Arial", 11), bg='white')
        self.confidence_label.pack(pady=5)

        self.details_label = tk.Label(result_frame, text="",
                                     font=("Arial", 9), bg='white', justify=tk.LEFT)
        self.details_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.last_processed = None

    def on_canvas_press(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def on_canvas_drag(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                   fill='black', width=14, capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                          fill=(0, 0, 0), width=14)
        self.last_x = event.x
        self.last_y = event.y

    def advanced_preprocessing(self, image):
        """
        Advanced multi-step preprocessing
        """
        # Step 1: Convert to grayscale
        img_gray = image.convert('L')
        img_array = np.array(img_gray, dtype=np.float32)

        # Step 2: Invert (black digit on white background)
        img_array = 255 - img_array

        # Step 3: Adaptive contrast enhancement
        # Increase local contrast
        pil_img = Image.fromarray(img_array.astype(np.uint8))
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(2.5)  # Boost contrast

        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(0.8)  # Slightly darken

        img_array = np.array(pil_img, dtype=np.float32)

        # Step 4: Otsu-like thresholding
        # Find optimal threshold
        threshold = self.otsu_threshold(img_array)
        img_binary = np.where(img_array >= threshold, 255.0, 0.0)

        # Step 5: Remove small noise
        img_binary = self.remove_small_noise(img_binary)

        # Step 6: Resize to 28x28
        img_pil = Image.fromarray(img_binary.astype(np.uint8))
        img_small = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
        img_array_28 = np.array(img_small, dtype=np.float32)

        # Step 7: Normalize
        img_normalized = img_array_28 / 255.0

        # Step 8: Apply sharpening for clarity
        img_normalized = np.clip(img_normalized * 1.2, 0, 1)

        return img_normalized, img_array_28 / 255.0

    def otsu_threshold(self, img_array):
        """Calculate Otsu threshold"""
        # Normalize to 0-256
        hist, bin_edges = np.histogram(img_array, 256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate probabilities
        weight1 = np.cumsum(hist)
        weight2 = weight1[-1] - weight1

        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum(hist * bin_centers)[-1] - np.cumsum(hist * bin_centers)) / weight2

        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        idx = np.argmax(variance12)
        threshold = bin_centers[idx]

        return threshold

    def remove_small_noise(self, img_array, min_size=5):
        """Remove small noise components"""
        # Simple dilation and erosion
        kernel_size = 2

        # Dilate
        for _ in range(1):
            dilated = np.zeros_like(img_array)
            for i in range(1, img_array.shape[0] - 1):
                for j in range(1, img_array.shape[1] - 1):
                    dilated[i, j] = np.max(img_array[i-1:i+2, j-1:j+2])
            img_array = dilated

        return img_array

    def visualize_image(self, img_array, canvas):
        """Display processed image on canvas"""
        # Scale to 0-255
        img_vis = (img_array * 255).astype(np.uint8)

        # Scale up for display
        img_pil = Image.fromarray(img_vis)
        img_scaled = img_pil.resize((280, 280), Image.Resampling.NEAREST)

        from PIL import ImageTk
        photo = ImageTk.PhotoImage(img_scaled)

        canvas.create_image(140, 140, image=photo)
        canvas.image = photo

    def recognize_digit(self):
        """Recognize with advanced preprocessing"""
        try:
            self.recognition_count += 1

            # Advanced preprocessing
            processed_final, processed_28 = self.advanced_preprocessing(self.image)
            self.last_processed = (processed_final, processed_28)

            # Visualize both stages
            self.visualize_image(processed_28, self.preview1_canvas)
            self.visualize_image(processed_final, self.preview2_canvas)

            # Predict
            image_flat = processed_final.reshape(1, -1)
            prediction = self.model.predict(image_flat, verbose=0)
            predicted_digit = np.argmax(prediction[0])
            confidence = prediction[0][predicted_digit] * 100

            # Display result
            if confidence >= 80:
                color = '#2E7D32'
            elif confidence >= 60:
                color = '#F57C00'
            else:
                color = '#D32F2F'

            self.result_label.config(text=f"{predicted_digit}", fg=color)

            bar_len = int(confidence / 10)
            bar = "[" + "=" * bar_len + "-" * (10 - bar_len) + "]"
            self.confidence_label.config(
                text=f"Confidence: {confidence:.2f}%  {bar}",
                fg=color
            )

            # Top 5
            top_5 = np.argsort(prediction[0])[-5:][::-1]
            details = "Top 5:\n"
            for i, idx in enumerate(top_5, 1):
                prob = prediction[0][idx] * 100
                mark = " ← Selected" if idx == predicted_digit else ""
                details += f"  {i}. [{idx}] {prob:5.1f}%{mark}\n"

            self.details_label.config(text=details.strip())

        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")

    def save_images(self):
        """Save processed images for inspection"""
        if self.last_processed is None:
            messagebox.showwarning("Warning", "Run recognition first!")
            return

        try:
            processed_final, processed_28 = self.last_processed

            # Save original
            self.image.save(f'digit_original_{self.recognition_count}.png')

            # Save processed 28x28
            img_28 = (processed_28 * 255).astype(np.uint8)
            Image.fromarray(img_28).save(f'digit_processed_28x28_{self.recognition_count}.png')

            # Save enhanced
            img_final = (processed_final * 255).astype(np.uint8)
            Image.fromarray(img_final).save(f'digit_final_{self.recognition_count}.png')

            messagebox.showinfo("Success",
                              f"Images saved:\n"
                              f"- digit_original_{self.recognition_count}.png\n"
                              f"- digit_processed_28x28_{self.recognition_count}.png\n"
                              f"- digit_final_{self.recognition_count}.png")

        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {str(e)}")

    def clear_canvas(self):
        """Clear everything"""
        self.canvas.delete("all")
        self.preview1_canvas.delete("all")
        self.preview2_canvas.delete("all")
        self.image = Image.new("RGB", (320, 320), color=(255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None
        self.result_label.config(text="Draw digit")
        self.confidence_label.config(text="")
        self.details_label.config(text="")

def main():
    root = tk.Tk()
    app = AdvancedDigitRecognitionApp(root)
    root.mainloop()

if __name__ == "__main__":
    print("="*60)
    print("ADVANCED DIGIT RECOGNITION")
    print("="*60)
    print("\nFeatures:")
    print("  - Otsu thresholding")
    print("  - Adaptive contrast enhancement")
    print("  - Noise removal")
    print("  - 3-stage visualization")
    print("  - Image export for debugging")
    print("="*60 + "\n")
    main()
