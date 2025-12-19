# Handwritten Digit & Letter Recognition System

## Project Overview

This project implements a handwritten digit and letter recognition system based on the EMNIST Balanced dataset. The system supports recognition of handwritten digits (0–9) 
and letters (A–Z, a–k) using both classical machine learning methods and deep learning models. A Streamlit-based graphical user interface (GUI) is provided for real-time 
user interaction, including drawing characters on a canvas or uploading images.

The project demonstrates a complete machine learning pipeline, including data preprocessing, model training, evaluation, and deployment in an interactive application.

---

## Dataset

* **Dataset:** EMNIST Balanced
* **Number of classes:** 47 (digits 0–9, uppercase A–Z, selected lowercase letters a–k)
* **Image size:** 28 × 28 grayscale

All input images are preprocessed to match the official EMNIST format, including binarization, bounding-box cropping, resizing, padding, and orientation correction.

---

## Models Implemented

The following models are implemented and compared:

1. **Logistic Regression (Pixels)**
   A baseline model trained directly on flattened pixel values.

2. **HOG + Logistic Regression**
   A classical machine learning approach using Histogram of Oriented Gradients (HOG) features combined with logistic regression.

3. **MyCNN**
   A custom convolutional neural network designed for handwritten character recognition.

4. **AdvancedCNN**
   A deeper convolutional neural network with additional convolutional layers and regularization to improve feature extraction and generalization.

These models form a progressive comparison chain from traditional methods to deep learning approaches.

---

## Training Details

* **Loss function:** Cross-entropy loss (CNN models)
* **Optimizer:** Adam
* **Training epochs:**

  * MyCNN: 10 epochs
  * AdvancedCNN: 12 epochs

The number of epochs was selected based on validation performance. Experiments showed that model accuracy converged around 10–12 epochs, and additional training did not lead to consistent improvements.

---

## Preprocessing Pipeline

The preprocessing steps are consistent across training and inference:

1. Convert input image to grayscale
2. Otsu thresholding with color inversion (black background, white foreground)
3. Bounding-box cropping
4. Aspect-ratio-preserving resize (longest side → 20 pixels)
5. Zero-padding to 28 × 28
6. EMNIST orientation correction (rotation and horizontal flip)

This ensures compatibility between user input images and the EMNIST dataset distribution.

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure trained model files are available in the project root directory:

* `cnn_model.pth`
* `advancedcnn_model.pth`
* `logreg_model.pkl`
* `logreg_hog.pkl`

3. Launch the application:

```bash
streamlit run app.py
```

4. Open the provided local URL in a web browser and start drawing or uploading handwritten characters.

---

## Model Evaluation

The application includes a model evaluation mode that displays:

* Overall accuracy for each model
* Learning curves (accuracy and loss)
* Confusion matrices
* Full classification reports

This allows a comprehensive comparison of classical and deep learning approaches.

---

## Notes and Observations

* Logistic Regression models show strong robustness for characters with stable global structures (e.g., M and P).
* CNN-based models achieve higher overall accuracy but may confuse structurally similar characters due to spatial normalization and pooling effects.
* This behavior highlights the trade-off between global feature preservation and local pattern learning.

---

## Technologies Used

* Python
* PyTorch
* Scikit-learn
* OpenCV
* Streamlit
* NumPy, Pandas

---


