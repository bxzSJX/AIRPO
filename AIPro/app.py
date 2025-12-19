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

# 假设这些模型和数据集工具已存在
from models.my_cnn import MyCNN
from models.advanced_cnn import AdvancedCNN

# fix_emnist_orientation, load_emnist 假定来自 dataset.py

# =========================
# Streamlit 全局设置 + 简单样式
# =========================

st.set_page_config(
    page_title="Handwritten Digit & Letter Recognition",
    page_icon="✏️",
    layout="centered"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* 背景 */
.stApp {
    background: linear-gradient(180deg, #f7f9fc 0%, #eef1f5 100%);
}

/* 主容器居中 & 宽度优化 */
.main {
    max-width: 900px;
    margin: 0 auto;
}

/* 标题美化 */
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

/* 玻璃拟态卡片效果 */
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

/* 侧边栏 */
section[data-testid="stSidebar"] {
    background-color: #ffffffee;
    backdrop-filter: blur(4px);
    padding-left: 10px;
}

/* 输入框美化 */
input, textarea {
    border-radius: 10px !important;
}

/* 下拉框、单选按钮文本 */
.stSelectbox label, .stRadio label {
    font-weight: 600;
    font-size: 15px;
}

/* 按钮 */
button[kind="secondary"] {
    border-radius: 12px !important;
}

/* 成功提示结果 */
.stAlert {
    border-radius: 12px;
    font-size: 1.15rem !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# EMNIST Balanced 映射表（官方 47 类）
# =========================
EMNIST_LABEL_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',

    # Lowercase — EMNIST Balanced official order
    36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g',
    43: 'h', 44: 'i', 45: 'j', 46: 'k'
}

# 前 10 个索引是数字，后面的是字母
DIGIT_IDX = list(range(10))  # 0-9
LETTER_IDX = list(range(10, 47))  # 10-46


# =========================
# 模型加载
# =========================

@st.cache_resource
def load_cnn():
    """加载训练好的 CNN（47 类 EMNIST Balanced）"""
    # 必须确保 cnn_model.pth 文件存在
    if not os.path.exists("cnn_model.pth"):
        st.error("模型文件 cnn_model.pth 未找到！请确保它在 app.py 同级目录下。")
        return None
    model = MyCNN(num_classes=47)
    state = torch.load("cnn_model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource
def load_logreg():
    """加载 Logistic Regression 基线模型"""
    # 必须确保 logreg_model.pkl 文件存在
    if not os.path.exists("logreg_model.pkl"):
        st.error("模型文件 logreg_model.pkl 未找到！请确保它在 app.py 同级目录下。")
        return None
    return joblib.load("logreg_model.pkl")


@st.cache_resource
def load_advancedcnn():
    """加载训练好的 Advanced CNN（47 类 EMNIST Balanced）"""
    # 必须确保 advancedcnn_model.pth 文件存在
    if not os.path.exists("advancedcnn_model.pth"):
        st.error("模型文件 advancedcnn_model.pth 未找到！请确保它在 app.py 同级目录下。")
        return None
    model = AdvancedCNN(num_classes=47)
    state = torch.load("advancedcnn_model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource
def load_logreg_hog():
    """加载 HOG + Logistic Regression 模型"""
    if not os.path.exists("logreg_hog.pkl"):
        st.error("模型文件 logreg_hog.pkl 未找到！")
        return None
    return joblib.load("logreg_hog.pkl")



# =========================
# 选择标签：自动/只数字/只字母/全类
# =========================

def choose_label_by_mode(probs: np.ndarray, mode: str) -> int:
    if mode == "Digits only":
        candidate_idx = DIGIT_IDX

    elif mode == "Letters only":
        candidate_idx = LETTER_IDX

    elif mode == "All 47 classes":
        return int(np.argmax(probs))

    else:
        # Auto: 看最高概率属于数字还是字母
        best = int(np.argmax(probs))
        # 10是A，所以 <10 是数字
        if best < 10:
            return best  # 0–9
        else:
            return best  # 10–46（字母）

    # 手动过滤范围
    candidate_probs = probs[candidate_idx]
    # np.argmax 返回的是在 candidate_probs 中的索引，需要映射回原始索引
    local_best = int(candidate_probs.argmax())
    return candidate_idx[local_best]


# =========================
# 图像预处理（与 dataset.py 保持一致的方向）
# =========================

def preprocess_image(gray_img: np.ndarray):
    """
    最终版：100% 复现 EMNIST 预处理流程：
    - 二值化（黑底白字）
    - bbox 裁剪
    - 等比缩放到 20x20
    - 28x28 padding 居中
    - EMNIST 旋转 + 镜像修正
    """

    # 1) uint8
    gray = gray_img.astype("uint8")

    # 2) OTSU + 反色 → 黑底白字
    _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3) bbox 裁剪
    coords = cv2.findNonZero(img)
    if coords is None:
        canvas = np.zeros((28, 28), dtype=np.uint8)
        tensor = torch.zeros((1, 1, 28, 28))
        return canvas, canvas, tensor, canvas.reshape(1, -1) / 255

    x, y, w, h = cv2.boundingRect(coords)
    crop = img[y:y+h, x:x+w]

    # 4) 等比缩放到最长边=20
    scale = 20 / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    small = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 5) padding 到 28×28
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = small

    # 6) ⭐⭐ EMNIST 方向修复（必须保留）
    canvas = np.rot90(canvas, 3)
    canvas = np.fliplr(canvas)

    # 输出两张图用于 GUI 显示
    img_fixed = canvas.copy()
    img_resized = cv2.resize(gray, (28, 28))

    # 转 tensor
    tensor = torch.tensor(canvas / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    flat = canvas.reshape(1, -1) / 255.0

    return img_resized, img_fixed, tensor, flat



# =========================
# 单张图像预测
# =========================

def predict_single(gray_img: np.ndarray, model_name: str, category_mode: str):
    """
    返回：img_resized, img_fixed, pred_idx, probs
    """

    img_resized, img_fixed, tensor, flat = preprocess_image(gray_img)
    model = None

    # ======================================================
    # 1) Logistic Regression
    # ======================================================
    if model_name == "Logistic Regression":
        clf = load_logreg()
        if clf is None: return img_resized, img_fixed, -1, None  # 无法加载模型

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(flat)[0]  # shape = (47,)
            pred_idx = choose_label_by_mode(probs, category_mode)
        else:
            pred_idx = int(clf.predict(flat)[0])
            probs = None

        return img_resized, img_fixed, pred_idx, probs

    # ======================================================
    # 2) MyCNN (你的基础 CNN)
    # ======================================================
    elif model_name == "MyCNN":
        model = load_cnn()

    # ======================================================
    # 3) Advanced CNN（高级 CNN）
    # ======================================================
    elif model_name == "AdvancedCNN":
        model = load_advancedcnn()

    elif model_name == "HOG_LR":
        clf = load_logreg_hog()
        if clf is None:
            return img_resized, img_fixed, -1, None

        # 使用 img_fixed 计算 HOG 特征（和训练一致）
        feat = hog(
            img_fixed,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        ).reshape(1, -1)

        # 获取概率
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(feat)[0]
            pred_idx = choose_label_by_mode(probs, category_mode)
        else:
            pred_idx = int(clf.predict(feat)[0])
            probs = None

        return img_resized, img_fixed, pred_idx, probs

    else:
        raise ValueError(f"未知模型名称: {model_name}")

    if model is None:
        return img_resized, img_fixed, -1, None  # 无法加载模型

    # ======================================================
    # 深度学习模型预测
    # ======================================================
    # ======================================================
    # 深度学习模型预测（⭐ 加入“多方向尝试”）
    # ======================================================

    # 所有候选方向
    candidates = []

    # 原始 tensor
    candidates.append(("orig", tensor))

    # rot90 / rot180 / rot270
    for k in [1, 2, 3]:
        rotated = torch.rot90(tensor, k, [2, 3])
        candidates.append((f"rot{k}", rotated))

    # flip（水平翻转）
    flip = torch.flip(tensor, [3])
    candidates.append(("flip", flip))

    # flip 后再 rot90 / rot180 / rot270
    for k in [1, 2, 3]:
        candidates.append((f"flip_rot{k}", torch.rot90(flip, k, [2, 3])))

    best_score = -1
    best_idx = None
    best_probs = None
    best_direction = None

    # 逐个方向尝试
    with torch.no_grad():
        for name, t in candidates:
            logits = model(t)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            # 选择对应模式下最好的类（digit-only / letter-only / all）
            idx = choose_label_by_mode(probs, category_mode)
            score = probs[idx]  # 该方向下的可信度

            if score > best_score:
                best_score = score
                best_idx = idx
                best_probs = probs
                best_direction = name

    # 👉 如果你想调试看看选了哪个方向，可以取消注释：
    # st.write(f"使用方向: {best_direction}")

    return img_resized, img_fixed, best_idx, best_probs


# =========================
# 侧边栏设置
# =========================

st.sidebar.title("⚙️ Settings")
mode = st.sidebar.radio(
    "选择模式 / Mode",
    ("单张识别 Single Image", "模型评估 Model Evaluation")
)

st.sidebar.markdown("---")  # 分隔线

model_choice = st.sidebar.radio(
    "选择模型 / Model",
    ("CNN (MyCNN)", "Advanced CNN", "Logistic Regression (Baseline)", "HOG + Logistic Regression")
)


st.sidebar.markdown("---")  # 分隔线

category_mode = st.sidebar.selectbox(
    "类别范围 / Category range",
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


# =========================
# 主标题
# =========================

st.markdown("""
<div style="text-align:center; margin-top:-20px;">
    <h1>🧠 Handwritten Digit & Letter Recognition</h1>
    <p style="color:#555; font-size:16px;">
        支持 0–9 与 A–Z/a–z 手写字符识别 · 基于 EMNIST Balanced · 实时预处理与模型推理
    </p>
</div>
""", unsafe_allow_html=True)


# =========================
# 模式 1：单张识别
# =========================

if mode.startswith("单张"):

    # ------ 1. 输入方式 ------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1. 输入方式 / Input")

    input_mode = st.radio(
        "选择输入方式：",
        ("上传图片 Upload Image", "画板手写 Draw on Canvas")
    )

    gray_img = None

    # 上传图片
    if input_mode.startswith("上传"):
        uploaded = st.file_uploader(
            "上传一张包含单个数字/字母的图片（任意大小，背景尽量简单）",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded is not None:
            # 确保转换为灰度图 'L'
            pil = Image.open(uploaded).convert("L")
            gray_img = np.array(pil)
            st.image(pil, caption="Original Image", use_container_width=True)

    # 画板手写
    else:
        st.write("在下面画一个数字或字母：")
        # 200x200 画布，白色背景，黑色笔迹
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
            # st_canvas 返回的是 RGBA
            img_rgba = canvas_result.image_data.astype("uint8")
            # 转换为灰度图
            gray_img = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)

    st.markdown('</div>', unsafe_allow_html=True)

    # ------ 2. 预处理 & 预测 ------
    if gray_img is not None:
        img_resized, img_fixed, pred_idx, probs = predict_single(
            gray_img, current_model, category_mode
        )

        # 如果模型加载失败
        if pred_idx == -1:
            st.warning("模型加载失败或未找到模型文件，请检查您的模型文件路径和名称。")
        else:
            # 2.1 显示预处理
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("2. 图像预处理 / Preprocessing")

            col1, col2 = st.columns(2)
            with col1:
                # 原始图像缩放版本 (for display only)
                st.image(img_resized, caption="Resize 28×28", width=150)
            with col2:
                # 经过 EMNIST 方向修正和黑白反转的版本 (Actual Input to Model)
                st.image(img_fixed, caption="Fixed Input", width=150)

            st.markdown('</div>', unsafe_allow_html=True)

            # 2.2 显示预测结果
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("3. 识别结果 / Prediction")

            char_label = EMNIST_LABEL_MAP.get(pred_idx, f"class {pred_idx}")
            st.success(f"预测类别: {pred_idx} | 字符: {char_label}")

            if probs is not None:
                topk = 5
                # 找到概率最高的 Top-5 索引
                top_idx = np.argsort(probs)[-topk:][::-1]
                st.write("Top-5 概率：")

                rows = []
                for rank, i in enumerate(top_idx, start=1):
                    label = EMNIST_LABEL_MAP.get(i, f"class {i}")
                    rows.append({
                        "Rank ": rank,
                        "类别 ID": int(i),
                        "字符 Label": label,
                        "概率 Probability": f"{probs[i]:.4f}",
                    })

                df_top5 = pd.DataFrame(rows)


                # 用 dataframe 展示，去掉左侧索引
                st.dataframe(df_top5, hide_index=True, use_container_width=True)

    else:
        st.info("请先上传图片或在画板上书写一个字符。")

# =========================
# 模式 2：模型评估
# =========================

else:
    # 模式评估部分依赖于本地的报告和图像文件 (如 .txt, .png)

    # =====================================
    #   1. Overall Metrics Summary
    # =====================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1. 模型整体性能 / Overall Metrics")

    # --- Logistic Regression (pixels) ----
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

    # ---- HOG + Logistic Regression ----
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

    # ---- CNN Accuracy ----
    cnn_acc = "N/A"
    if os.path.exists("cnn_test_accuracy.txt"):
        with open("cnn_test_accuracy.txt", "r") as f:
            try:
                v = float(f.read().strip())
                cnn_acc = f"{v * 100:.2f}%"
            except:
                pass

    # ---- Advanced CNN Accuracy ----
    adv_acc = "N/A"
    if os.path.exists("advancedcnn_test_accuracy.txt"):
        with open("advancedcnn_test_accuracy.txt", "r") as f:
            try:
                v = float(f.read().strip())
                adv_acc = f"{v * 100:.2f}%"
            except:
                pass

    st.write(f"**Logistic Regression (Pixels):** baseline，准确率约 **{lr_pixels_acc}**")
    st.write(f"**HOG + Logistic Regression:** 使用人工特征后提升，准确率约 **{hog_acc_str}**")
    st.write(f"**CNN (Deep Learning):** 自定义卷积网络，测试准确率约 **{cnn_acc}**")
    st.write(f"**Advanced CNN:** 更深的自定义卷积网络，测试准确率约 **{adv_acc}**")

    st.info("模型比较链：Logistic Regression（像素） → HOG+LR（特征工程） → CNN（深度学习） → Advanced CNN（更深模型）")
    st.markdown('</div>', unsafe_allow_html=True)

    # =====================================
    #   2. CNN Learning Curves
    # =====================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2. CNN 学习曲线 / Learning Curves")

    if os.path.exists("learning_curve.png"):
        st.image("learning_curve.png", caption="CNN Learning Curve", use_container_width=True)
    if os.path.exists("loss_curve.png"):
        st.image("loss_curve.png", caption="CNN Loss Curve", use_container_width=True)
    if not os.path.exists("learning_curve.png") and not os.path.exists("loss_curve.png"):
        st.info("未找到学习曲线图片 (learning_curve.png / loss_curve.png)")

    st.markdown('</div>', unsafe_allow_html=True)

    # =====================================
    #   3. Confusion Matrices (ALL MODELS)
    # =====================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("3. 各模型混淆矩阵 / Confusion Matrices")

    # ---- CNN ----
    st.markdown("### 🔵 CNN Confusion Matrix")
    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png", caption="CNN Confusion Matrix", use_container_width=True)
    else:
        st.info("未找到 CNN 混淆矩阵（confusion_matrix.png）")

    # ---- Advanced CNN ----
    st.markdown("### 🟠 Advanced CNN Confusion Matrix")
    if os.path.exists("confusion_matrix_advancedcnn.png"):
        st.image("confusion_matrix_advancedcnn.png",
                 caption="Advanced CNN Confusion Matrix",
                 use_container_width=True)
    else:
        st.info("未找到 Advanced CNN 混淆矩阵（confusion_matrix_advancedcnn.png），请先运行 evaluate_advancedcnn_confusion.py。")

    # ---- HOG + LR ----
    st.markdown("### 🟢 HOG + Logistic Regression Confusion Matrix")
    if os.path.exists("confusion_matrix_hog.png"):
        st.image("confusion_matrix_hog.png", caption="HOG + LR Confusion Matrix", use_container_width=True)
    else:
        st.info("未找到 HOG + LR 混淆矩阵（confusion_matrix_hog.png）")

    # ---- Logistic Regression (Pixels) ----
    st.markdown("### ⚪ Logistic Regression (Pixels) Confusion Matrix")
    if os.path.exists("confusion_matrix_logreg.png"):
        st.image("confusion_matrix_logreg.png", caption="LR (Pixels) Confusion Matrix", use_container_width=True)
    else:
        st.info("未找到 LR (Pixels) 混淆矩阵，请先运行 evaluate_logreg_pixels.py")

    st.markdown('</div>', unsafe_allow_html=True)

    # =====================================
    #   4. Full Classification Reports
    # =====================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("4. 分类报告 / Classification Reports")

    # --- LR Pixels ---
    st.markdown("### ⚪ Logistic Regression (Pixels) Report")
    try:
        with open("logreg_pixels_classification_report.txt", "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    except:
        st.info("未找到 logreg_pixels_classification_report.txt")

    # --- HOG + LR ---
    st.markdown("### 🟢 HOG + Logistic Regression Report")
    try:
        with open("logreg_hog_classification_report.txt", "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    except:
        st.info("未找到 logreg_hog_classification_report.txt")

    # --- CNN ---
    st.markdown("### 🔵 CNN Classification Report")
    try:
        with open("cnn_classification_report.txt", "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    except:
        st.info("未找到 cnn_classification_report.txt，请先运行 evaluate_cnn_report.py")

    # --- Advanced CNN ---
    st.markdown("### 🟠 Advanced CNN Classification Report")
    try:
        with open("advancedcnn_classification_report.txt", "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    except:
        st.info("未找到 advancedcnn_classification_report.txt，请先运行 evaluate_advancedcnn_confusion.py")

    st.markdown('</div>', unsafe_allow_html=True)

    # =====================================
    #   5. Model Comparison Chart
    # =====================================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("5. 模型总体对比 / Model Comparison")

    if os.path.exists("model_comparison.png"):
        st.image("model_comparison.png", caption="Model Comparison", use_container_width=True)
    else:
        st.info("请先运行 generate_model_comparison.py 生成 model_comparison.png")

    st.markdown('</div>', unsafe_allow_html=True)
