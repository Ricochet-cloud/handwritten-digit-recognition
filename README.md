# 🖊️ Handwritten Digit Recognition

A comprehensive handwritten digit recognition system using deep learning models. Includes multiple versions from basic neural networks to advanced CNN models with 99.4% accuracy.

## 🎯 Features

- **Multiple Models**
  - Dense Neural Network (97.9% accuracy)
  - CNN Model (99.4% accuracy) ⭐ **Recommended**

- **5 Interactive Versions**
  - Basic version with simple GUI
  - Improved version with better preprocessing
  - Debug version with image visualization
  - Advanced version with Otsu thresholding
  - CNN version with highest accuracy

- **User-Friendly Interface**
  - Easy mouse drawing on canvas
  - Real-time digit recognition
  - Confidence scores and top-5 predictions
  - Windows batch file launchers

## 📊 Model Performance

| Model | Accuracy | Type |
|-------|----------|------|
| Dense Network | 97.9% | Basic |
| CNN | 99.4% | Advanced ⭐ |

Both trained on MNIST dataset (60,000 training samples, 10,000 test samples)

## 🚀 Quick Start

### Option 1: GUI Application (Recommended)

**Windows Users:**
- Double-click: `숫자인식_CNN_최고정확도_실행.bat`

**Mac/Linux Users:**
```bash
python draw_cnn_recognition.py
```

### Option 2: Train CNN Model from Scratch

```bash
python train_cnn_model.py
```

This will:
1. Download MNIST dataset (11.5 MB)
2. Train CNN model for 15 epochs
3. Achieve 99.4% test accuracy
4. Save model as `digit_cnn_model.h5`

### Option 3: Python Scripts

```bash
# Train basic Dense model
python digit_recognition_simple.py

# Interactive recognition
python interactive_digit_recognition.py

# Advanced debug mode
python draw_and_recognize_debug.py

# CNN high-accuracy version
python draw_cnn_recognition.py
```

## 📋 File Structure

```
.
├── README.md
├── train_cnn_model.py              # CNN model trainer
├── draw_cnn_recognition.py          # CNN GUI app (99.4% accuracy)
├── draw_and_recognize_advanced.py   # Advanced preprocessing
├── draw_and_recognize_debug.py      # Debug mode with visualization
├── draw_and_recognize_improved.py   # Improved preprocessing
├── digit_recognition_simple.py      # Basic training & testing
├── interactive_digit_recognition.py # Interactive mode
├── digit_model.h5                   # Trained Dense model (97.9%)
├── digit_cnn_model.h5               # Trained CNN model (99.4%)
├── 숫자인식_CNN_최고정확도_실행.bat   # CNN launcher (Windows)
├── 숫자인식_고급버전_실행.bat        # Advanced launcher (Windows)
├── 숫자인식_디버그버전_실행.bat      # Debug launcher (Windows)
├── 숫자인식_개선버전_실행.bat        # Improved launcher (Windows)
└── 숫자인식_실행.bat                # Basic launcher (Windows)
```

## 🔧 Requirements

```
Python 3.7+
numpy
tensorflow (with keras)
Pillow (PIL)
tkinter (usually comes with Python)
```

### Installation

```bash
pip install numpy tensorflow pillow
```

## 💡 Usage Tips

### For Best Recognition:

1. **Draw Large** - Make digit fill 70% of canvas
2. **Draw Dark** - Use thick, dark strokes
3. **Center It** - Place digit in middle of canvas
4. **Be Slow** - Draw smoothly, not too fast
5. **Complete** - Draw full digit without gaps

### Image Preview

In debug mode, you can see:
- **Left**: Original drawing (400x400)
- **Middle**: Processed image (28x28)
- **Right**: Final enhanced image for model

## 🏗️ CNN Architecture

```
Input: 28x28x1 grayscale image
    ↓
Conv2D(32, 3x3) → ReLU → MaxPool(2x2) → Dropout(0.25)
    ↓
Conv2D(64, 3x3) → ReLU → MaxPool(2x2) → Dropout(0.25)
    ↓
Conv2D(128, 3x3) → ReLU → Dropout(0.25)
    ↓
Flatten
    ↓
Dense(256) → ReLU → Dropout(0.5)
    ↓
Dense(128) → ReLU → Dropout(0.5)
    ↓
Dense(10) → Softmax (output: digits 0-9)
```

## 📈 Training Results

### CNN Model Training (15 epochs):
- Final Training Accuracy: 99.15%
- Final Validation Accuracy: 99.37%
- Test Accuracy: 99.40%
- Loss: 0.0213

### Test Results on All Digits:
- Digit 0-9: ~99-100% accuracy each

## 🔍 Image Preprocessing Steps

1. Convert to grayscale
2. Invert colors (white background)
3. Enhance contrast (2.0x)
4. Resize to 28x28
5. Normalize to [0, 1]
6. Reshape for CNN: (28, 28, 1)

## 🐛 Troubleshooting

**Problem:** Low recognition accuracy (~60%)
- **Solution:** Use CNN version (99.4%) instead of Dense model

**Problem:** GUI doesn't appear
- **Solution:** Ensure tkinter is installed: `pip install tk`

**Problem:** Model file not found
- **Solution:** Run `python train_cnn_model.py` to train and save the model

**Problem:** scipy not installed (for old debug version)
- **Solution:** Use newer versions without scipy dependency

## 📚 Dataset Information

- **Dataset**: MNIST (Modified National Institute of Standards and Technology)
- **Training Samples**: 60,000
- **Test Samples**: 10,000
- **Image Size**: 28x28 pixels
- **Format**: Grayscale

## 🎓 Educational Value

This project demonstrates:
- Deep Learning fundamentals
- CNN architecture and training
- Image preprocessing techniques
- GUI programming with tkinter
- Model evaluation and metrics
- Batch prediction systems

## 📄 License

MIT License - Feel free to use for educational purposes

## 🤝 Contributing

Improvements welcome! Some ideas:
- Web interface (Flask/Django)
- Real-time webcam digit recognition
- Multi-digit number recognition
- Drawing to file export
- Model optimization (quantization)

## 📞 Contact

Feel free to fork, modify, and share!

---

**Made with ❤️ for digit recognition enthusiasts**
