import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# ======================
# CONFIG
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["glass", "metal", "paper", "plastic"]
MODEL_PATH = "waste_model.pth"

# ======================
# WASTE INFO (TR / EN)
# ======================
WASTE_INFO = {
    "plastic": {
        "en": "Recycle separately. Never burn. Takes centuries to decompose.",
        "tr": "Ayrı toplanmalıdır. Asla yakılmamalıdır. Yüzyıllarca doğada kalır."
    },
    "paper": {
        "en": "Recycle clean paper only. Saves trees and water.",
        "tr": "Temiz kağıtlar geri dönüştürülmelidir. Ağaçları ve suyu korur."
    },
    "glass": {
        "en": "100% recyclable. Separate by color if possible.",
        "tr": "Yüzde 100 geri dönüştürülebilir. Mümkünse rengine göre ayırın."
    },
    "metal": {
        "en": "Recycle aluminum and steel. Saves energy and resources.",
        "tr": "Alüminyum ve çelik geri dönüştürülmelidir. Enerji tasarrufu sağlar."
    }
}

# ======================
# MODEL
# ======================
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, len(CLASS_NAMES))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ======================
# TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ======================
# GRAD-CAM
# ======================
def gradcam(model, x, class_idx):
    activations = []
    gradients = []

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0])

    layer = model.features[-1]
    h1 = layer.register_forward_hook(forward_hook)
    h2 = layer.register_full_backward_hook(backward_hook)

    output = model(x)
    model.zero_grad()
    output[0, class_idx].backward()

    h1.remove()
    h2.remove()

    A = activations[0].detach().cpu().numpy()[0]
    G = gradients[0].detach().cpu().numpy()[0]

    weights = G.mean(axis=(1, 2))
    cam = np.zeros(A.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * A[i]

    cam = np.maximum(cam, 0)
    cam /= cam.max() + 1e-8
    return cam

# ======================
# UI
# ======================
st.set_page_config(layout="wide")
st.title("♻️ AI-Based Waste Classification with Grad-CAM")

# -------- LANGUAGE SELECT --------
lang = st.sidebar.radio("🌍 Language / Dil", ["EN", "TR"]).lower()

tab1, tab2 = st.tabs(["🖼 Image Upload Grad-CAM", "🎥 Live Webcam Grad-CAM"])

# ======================
# IMAGE UPLOAD
# ======================
with tab1:
    uploaded = st.file_uploader(
        "Upload an image / Görsel yükle",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)

        x = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
            pred = int(probs.argmax())
            conf = float(probs[0][pred])

        class_name = CLASS_NAMES[pred]

        st.success(f"Prediction: **{class_name.upper()}**")
        st.write(f"Confidence: **{conf:.2f}**")

        st.markdown("### 🗑 Waste Information")
        st.info(WASTE_INFO[class_name][lang])

        # -------- GRAD-CAM CONTROLS --------
        st.markdown("### 🔥 Grad-CAM Settings")

        alpha = st.slider("Overlay Intensity", 0.0, 1.0, 0.5)
        cmap_name = st.selectbox(
            "Color Map",
            ["JET", "HOT", "TURBO", "OCEAN"]
        )

        cmap_dict = {
            "JET": cv2.COLORMAP_JET,
            "HOT": cv2.COLORMAP_HOT,
            "TURBO": cv2.COLORMAP_TURBO,
            "OCEAN": cv2.COLORMAP_OCEAN
        }

        cam = gradcam(model, x, pred)
        cam = cv2.resize(cam, image.size)

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam),
            cmap_dict[cmap_name]
        )

        img_np = np.array(image)
        overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)

        st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

# ======================
# LIVE CAMERA
# ======================
with tab2:
    st.subheader("🎥 Live Webcam Grad-CAM")

    start = st.button("▶ Start Camera")
    stop = st.button("⏹ Stop Camera")

    if "run_cam" not in st.session_state:
        st.session_state.run_cam = False

    if start:
        st.session_state.run_cam = True
    if stop:
        st.session_state.run_cam = False

    frame_holder = st.image([])

    if st.session_state.run_cam:
        cap = cv2.VideoCapture(0)

        while st.session_state.run_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera error")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            x = transform(pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)
                pred = int(probs.argmax())
                conf = float(probs[0][pred])

            label = f"{CLASS_NAMES[pred]} | {conf:.2f}"
            cv2.putText(
                rgb,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            frame_holder.image(rgb)

        cap.release()
