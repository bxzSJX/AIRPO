import os
import numpy as np
import cv2
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn.functional as F
from torchvision import transforms
import joblib
import re
from skimage.feature import hog
import pandas as pd
import html
from models.my_cnn import MyCNN
from models.advanced_cnn import AdvancedCNN



# Streamlit Global settings
st.set_page_config(
    page_title="Handwritten Digit & Letter Recognition",
    layout="centered"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* background */
.stApp {
    background: linear-gradient(180deg, #f7f9fc 0%, #eef1f5 100%);
}

.main {
    max-width: 900px;
    margin: 0 auto;
}

h1 {
    font-weight: 800 !important;
    letter-spacing: -0.5px;
    text-align: center;
    margin-top: -10px;
    color: #222;
}

h2 {
    font-weight: 700 !important;
    margin-bottom: 0.8rem;
    color: #333;
}

.card {
    background: rgba(255, 255, 255, 0.75);
    padding: 1.6rem 2rem;
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    margin-bottom: 1.8rem;
    backdrop-filter: blur(10px);
    transition: 0.25s ease;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 28px rgba(0,0,0,0.1);
}

section[data-testid="stSidebar"] {
    background-color: #ffffffee;
    backdrop-filter: blur(4px);
    padding-left: 10px;
}

input, textarea {
    border-radius: 10px !important;
}

.stSelectbox label, .stRadio label {
    font-weight: 600;
    font-size: 15px;
}

button[kind="secondary"] {
    border-radius: 12px !important;
}

.stAlert {
    border-radius: 12px;
    font-size: 1.15rem !important;
}

</style>
""", unsafe_allow_html=True)


# EMNIST Balanced Mapping Tables (Official Class 47)

EMNIST_LABEL_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',

    # Lowercase — EMNIST Balanced official order
    36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g',
    43: 'h', 44: 'i', 45: 'j', 46: 'k'
}

# The first 10 indices are numbers, and the rest are letters.
DIGIT_IDX = list(range(10))  # 0-9
LETTER_IDX = list(range(10, 47))  # 10-46


@st.cache_resource
def load_cnn():
    if not os.path.exists("cnn_model.pth"):
        st.error(" cnn_model.pth Not found.")
        return None
    model = MyCNN(num_classes=47)
    state = torch.load("cnn_model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource
def load_logreg():
    if not os.path.exists("logreg_model.pkl"):
        st.error("logreg_model.pkl Not found.")
        return None
    return joblib.load("logreg_model.pkl")


@st.cache_resource
def load_advancedcnn():
    if not os.path.exists("advancedcnn_model.pth"):
        st.error("advancedcnn_model.pth Not found.")
        return None
    model = AdvancedCNN(num_classes=47)
    state = torch.load("advancedcnn_model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource
def load_logreg_hog():
    if not os.path.exists("logreg_hog.pkl"):
        st.error("logreg_hog.pkl Not found.")
        return None
    return joblib.load("logreg_hog.pkl")


# choose labels：Automatic / Numbers Only / Letters Only / All classes
def choose_label_by_mode(probs: np.ndarray, mode: str) -> int:
    if mode == "Digits only":
        candidate_idx = DIGIT_IDX

    elif mode == "Letters only":
        candidate_idx = LETTER_IDX

    elif mode == "All 47 classes":
        return int(np.argmax(probs))

    else:
        best = int(np.argmax(probs))
        if best < 10:
            return best
        else:
            return best

    candidate_probs = probs[candidate_idx]
    local_best = int(candidate_probs.argmax())
    return candidate_idx[local_best]



# Image preprocessing
def preprocess_image(gray_img: np.ndarray):
    # uint8
    gray = gray_img.astype("uint8")

    # 2)OTSU + Invert colors → Black background with white text
    _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3) bbox cropping
    coords = cv2.findNonZero(img)
    if coords is None:
        canvas = np.zeros((28, 28), dtype=np.uint8)
        tensor = torch.zeros((1, 1, 28, 28))
        return canvas, canvas, tensor, canvas.reshape(1, -1) / 255

    x, y, w, h = cv2.boundingRect(coords)
    crop = img[y:y+h, x:x+w]

    # 4) Scale proportionally to the longest side = 20
    scale = 20 / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    small = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 5) padding to 28×28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = small

    # 6) EMNIST Direction correction
    canvas = np.rot90(canvas, 3)
    canvas = np.fliplr(canvas)

    img_fixed = canvas.copy()
    img_resized = cv2.resize(gray, (28, 28))

    # Convert to tensor
    tensor = torch.tensor(canvas / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    flat = canvas.reshape(1, -1) / 255.0

    return img_resized, img_fixed, tensor, flat

def predict_single(gray_img: np.ndarray, model_name: str, category_mode: str):
    img_resized, img_fixed, tensor, flat = preprocess_image(gray_img)
    model = None

    # 1) Logistic Regression
    if model_name == "Logistic Regression":
        clf = load_logreg()
        if clf is None: return img_resized, img_fixed, -1, None

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(flat)[0]
            pred_idx = choose_label_by_mode(probs, category_mode)
        else:
            pred_idx = int(clf.predict(flat)[0])
            probs = None

        return img_resized, img_fixed, pred_idx, probs

    # 2) MyCNN

    elif model_name == "MyCNN":
        model = load_cnn()

    # 3) Advanced CNN
    elif model_name == "AdvancedCNN":
        model = load_advancedcnn()

    elif model_name == "HOG_LR":
        clf = load_logreg_hog()
        if clf is None:
            return img_resized, img_fixed, -1, None

        # Use `img_fixed` to calculate HOG features (consistent with training).
        feat = hog(
            img_fixed,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        ).reshape(1, -1)

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(feat)[0]
            pred_idx = choose_label_by_mode(probs, category_mode)
        else:
            pred_idx = int(clf.predict(feat)[0])
            probs = None

        return img_resized, img_fixed, pred_idx, probs

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if model is None:
        return img_resized, img_fixed, -1, None


    # Deep learning model prediction
    candidates = []
    candidates.append(("orig", tensor))

    # rot90 / rot180 / rot270
    for k in [1, 2, 3]:
        rotated = torch.rot90(tensor, k, [2, 3])
        candidates.append((f"rot{k}", rotated))

    # flip
    flip = torch.flip(tensor, [3])
    candidates.append(("flip", flip))

    # after flip  rot90 / rot180 / rot270
    for k in [1, 2, 3]:
        candidates.append((f"flip_rot{k}", torch.rot90(flip, k, [2, 3])))

    best_score = -1
    best_idx = None
    best_probs = None
    best_direction = None

    # try different directions
    with torch.no_grad():
        for name, t in candidates:
            logits = model(t)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            # Choose the best class under the corresponding pattern.
            idx = choose_label_by_mode(probs, category_mode)
            score = probs[idx]

            if score > best_score:
                best_score = score
                best_idx = idx
                best_probs = probs
                best_direction = name
    return img_resized, img_fixed, best_idx, best_probs


st.sidebar.title("Settings")
mode = st.sidebar.radio(
    "Mode Selection",
    ("Single Image", "Model Evaluation")
)

st.sidebar.markdown("---")

model_choice = st.sidebar.radio(
    "Choose Model",
    ("Logistic Regression (Baseline)", "HOG + Logistic Regression","CNN (MyCNN)" ,"Advanced CNN")
)

st.sidebar.markdown("---")

category_mode = st.sidebar.selectbox(
    "Category range",
    ("Auto (Digit vs Letter)", "Digits only", "Letters only", "All 47 classes")
)

if model_choice == "Logistic Regression (Baseline)":
    current_model = "Logistic Regression"
elif model_choice == "CNN (MyCNN)":
    current_model = "MyCNN"
elif model_choice == "Advanced CNN":
    current_model = "AdvancedCNN"
elif model_choice == "HOG + Logistic Regression":
    current_model = "HOG_LR"


st.markdown("""
<div style="text-align:center; margin-top:-20px;">
    <h1>Handwritten Digit & Letter Recognition</h1>
    <p style="color:#555; font-size:16px;">
        Draw or upload a character to get real-time predictions and Top-5 probabilities (EMNIST Balanced)
    </p>
</div>
""", unsafe_allow_html=True)



# Single recognize
if mode.startswith("Single Image"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1. Input")

    input_mode = st.radio(
        "Choose Input Ways",
        ("Upload Image", "Draw on Canvas")
    )
    gray_img = None

    if input_mode.startswith("Upload Image"):
        uploaded = st.file_uploader(
            "Upload an image containing a single number/letter (any size, simple background possible).",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded is not None:
            pil = Image.open(uploaded).convert("L")
            gray_img = np.array(pil)
            st.image(pil, caption="Original Image", use_container_width=True)


    else:
        st.write("Draw a number or letter below:")
        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_color="#000000",
            background_color="#FFFFFF",
            stroke_width=10,
            width=200,
            height=200,
            drawing_mode="freedraw",
            key="canvas",
        )
        if canvas_result.image_data is not None:
            # st_canvas returns RGBA
            img_rgba = canvas_result.image_data.astype("uint8")
            # Convert to grayscale
            gray_img = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)

    st.markdown('</div>', unsafe_allow_html=True)

    #  Preprocessing & predict
    if gray_img is not None:
        img_resized, img_fixed, pred_idx, probs = predict_single(
            gray_img, current_model, category_mode
        )

        if pred_idx == -1:
            st.warning("The model file was not found, please check your model file path and name.")
        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("2. Preprocessing")

            col1, col2 = st.columns(2)
            with col1:
                # for display only
                st.image(img_resized, caption="Resize 28×28", width=150)
            with col2:
                # Actual Input to Model
                st.image(img_fixed, caption="Fixed Input", width=150)

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("3. Prediction")

            char_label = EMNIST_LABEL_MAP.get(pred_idx, f"class {pred_idx}")
            st.success(f"Prediction Category: {pred_idx} | character: {char_label}")

            if probs is not None:
                topk = 5
                top_idx = np.argsort(probs)[-topk:][::-1]
                st.write("Top-5 probability：")

                rows = []
                for rank, i in enumerate(top_idx, start=1):
                    label = EMNIST_LABEL_MAP.get(i, f"class {i}")
                    rows.append({
                        "Rank ": rank,
                        "Category ID": int(i),
                        "Character Label": label,
                        "Probability": f"{probs[i]:.4f}",
                    })

                df_top5 = pd.DataFrame(rows)
                st.dataframe(df_top5, hide_index=True, use_container_width=True)

    else:
        st.info("Please upload an image or write a character on the canvas first.")


# model evaluation


else:

    #   1. Overall Metrics Summary
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1. Overall Metrics")

    # Logistic Regression (pixels)
    lr_pixels_acc = "N/A"
    if os.path.exists("logreg_pixels_classification_report.txt"):
        try:
            with open("logreg_pixels_classification_report.txt", "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            m = re.search(r"Accuracy:\s*([0-9.]+)", first_line)
            if m:
                lr_pixels_acc = f"{float(m.group(1)) * 100:.2f}%"
        except:
            pass

    # HOG + Logistic Regression
    hog_acc_str = "N/A"
    if os.path.exists("logreg_hog_classification_report.txt"):
        try:
            with open("logreg_hog_classification_report.txt", "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
            m = re.search(r"Accuracy:\s*([0-9.]+)", first_line)
            if m:
                hog_acc_str = f"{float(m.group(1)) * 100:.2f}%"
        except:
            pass

    # CNN Accuracy
    cnn_acc = "N/A"
    if os.path.exists("cnn_test_accuracy.txt"):
        with open("cnn_test_accuracy.txt", "r") as f:
            try:
                v = float(f.read().strip())
                cnn_acc = f"{v * 100:.2f}%"
            except:
                pass

    # Advanced CNN Accuracy
    adv_acc = "N/A"
    if os.path.exists("advancedcnn_test_accuracy.txt"):
        with open("advancedcnn_test_accuracy.txt", "r") as f:
            try:
                v = float(f.read().strip())
                adv_acc = f"{v * 100:.2f}%"
            except:
                pass

    st.write(f"**Logistic Regression (Pixels):** baseline，Accuracy rate approximately **{lr_pixels_acc}**")
    st.write(f"**HOG + Logistic Regression:** After using artificial features to improve accuracy, the accuracy rate was approximately [missing information]. **{hog_acc_str}**")
    st.write(f"**CNN (Deep Learning):** Custom convolutional network, test accuracy approximately **{cnn_acc}**")
    st.write(f"**Advanced CNN:** Deeper custom convolutional networks, with test accuracy approximately **{adv_acc}**")

    st.info("Model comparison chain: Logistic Regression (pixels) → HOG+LR (feature engineering) → CNN (deep learning) → Advanced CNN (deeper models)")
    st.markdown('</div>', unsafe_allow_html=True)


    #   2. CNN Learning Curves

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2. CNN Learning Curves")

    if os.path.exists("learning_curve.png"):
        st.image("learning_curve.png", caption="CNN Learning Curve", use_container_width=True)
    if os.path.exists("loss_curve.png"):
        st.image("loss_curve.png", caption="CNN Loss Curve", use_container_width=True)
    if not os.path.exists("learning_curve.png") and not os.path.exists("loss_curve.png"):
        st.info("Not found (learning_curve.png / loss_curve.png)")

    st.markdown('</div>', unsafe_allow_html=True)


    #   3. Confusion Matrices (ALL MODELS)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("3. Confusion Matrices")

    # ---- CNN ----
    st.markdown("### 🔵 CNN Confusion Matrix")
    if os.path.exists("confusion_matrix_cnn.png"):
        st.image("confusion_matrix.png", caption="CNN Confusion Matrix", use_container_width=True)
    else:
        st.info("Not found（confusion_matrix.png）")

    # ---- Advanced CNN ----
    st.markdown("### 🟠 Advanced CNN Confusion Matrix")
    if os.path.exists("confusion_matrix_advancedcnn.png"):
        st.image("confusion_matrix_advancedcnn.png",
                 caption="Advanced CNN Confusion Matrix",
                 use_container_width=True)
    else:
        st.info("No Advanced CNN confusion matrix found（confusion_matrix_advancedcnn.png），Please run it first evaluate_advancedcnn_confusion.py。")

    # ---- HOG + LR ----
    st.markdown("### 🟢 HOG + Logistic Regression Confusion Matrix")
    if os.path.exists("confusion_matrix_hog.png"):
        st.image("confusion_matrix_hog.png", caption="HOG + LR Confusion Matrix", use_container_width=True)
    else:
        st.info("No HOG + LR confusion matrix found（confusion_matrix_hog.png）")

    # ---- Logistic Regression (Pixels) ----
    st.markdown("### ⚪ Logistic Regression (Pixels) Confusion Matrix")
    if os.path.exists("confusion_matrix_logreg.png"):
        st.image("confusion_matrix_logreg.png", caption="LR (Pixels) Confusion Matrix", use_container_width=True)
    else:
        st.info("The LR (Pixels) confusion matrix was not found. Please run the command first evaluate_logreg_pixels.py")

    st.markdown('</div>', unsafe_allow_html=True)


    #   4. Full Classification Reports
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("4. Classification Reports")

    # --- LR Pixels ---
    st.markdown("### ⚪ Logistic Regression (Pixels) Report")
    try:
        with open("logreg_pixels_classification_report.txt", "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    except:
        st.info("Not found logreg_pixels_classification_report.txt")

    # --- HOG + LR ---
    st.markdown("### 🟢 HOG + Logistic Regression Report")
    try:
        with open("logreg_hog_classification_report.txt", "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    except:
        st.info("Not found logreg_hog_classification_report.txt")

    # --- CNN ---
    st.markdown("### 🔵 CNN Classification Report")
    try:
        with open("cnn_classification_report.txt", "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    except:
        st.info("Not found cnn_classification_report.txt，Please run it evaluate_cnn_report.py")

    # --- Advanced CNN ---
    st.markdown("### 🟠 Advanced CNN Classification Report")
    try:
        with open("advancedcnn_classification_report.txt", "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    except:
        st.info("Not found advancedcnn_classification_report.txt，run it evaluate_advancedcnn_confusion.py")

    st.markdown('</div>', unsafe_allow_html=True)


    #   5. Model Comparison Chart

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("5. Model Comparison")

    if os.path.exists("model_comparison.png"):
        st.image("model_comparison.png", caption="Model Comparison", use_container_width=True)
    else:
        st.info("run it generate_model_comparison.py generate model_comparison.png")

    st.markdown('</div>', unsafe_allow_html=True)
