import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import io

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = r"D:\Documents\HK2 2025-26 KHDL (Y3) Ky 6\BigData\waste-classification\models\resnet50_waste.pth"
MAP_PATH    = r"D:\Documents\HK2 2025-26 KHDL (Y3) Ky 6\BigData\waste-classification\models\class_mapping.json"

@st.cache_resource
def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, 2))
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval().to(DEVICE)
    return m, ckpt["class_to_idx"]

with open(MAP_PATH, encoding="utf-8") as f:
    CLASS_MAP = json.load(f)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(model, img_pil):
    tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return probs

def get_gradcam(model, img_pil, class_idx):
    tensor   = transform(img_pil).unsqueeze(0).to(DEVICE)
    cam      = GradCAM(model=model, target_layers=[model.layer4[-1]])
    targets  = [ClassifierOutputTarget(class_idx)]
    mask     = cam(input_tensor=tensor, targets=targets)[0]
    img_np   = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
    result   = show_cam_on_image(img_np, mask, use_rgb=True)
    return result

# --- UI ---
st.set_page_config(page_title="Phân loại rác thải", page_icon="♻️", layout="centered")
st.title("♻️ Phân loại rác thải")
st.caption("ResNet50 · Organic vs Recyclable")

uploaded = st.file_uploader("Upload ảnh rác thải", type=["jpg", "jpeg", "png"])

if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")
    model, class_to_idx = load_model()
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    probs     = predict(model, img_pil)
    pred_idx  = int(np.argmax(probs))
    pred_key  = idx_to_class[pred_idx]
    pred_name = CLASS_MAP[pred_key]
    cam_img   = get_gradcam(model, img_pil, pred_idx)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh gốc")
        st.image(img_pil, use_container_width=True)
    with col2:
        st.subheader("Grad-CAM")
        st.image(cam_img, use_container_width=True)

    st.markdown(f"### Kết quả: **{pred_name}** ({pred_key})")
    st.progress(float(probs[pred_idx]))

    fig, ax = plt.subplots(figsize=(5, 2.5))
    labels  = [CLASS_MAP[idx_to_class[i]] for i in range(2)]
    colors  = ["#2ecc71" if i == pred_idx else "#95a5a6" for i in range(2)]
    ax.barh(labels, probs * 100, color=colors)
    ax.set_xlabel("Xác suất (%)")
    ax.set_xlim(0, 100)
    for i, v in enumerate(probs):
        ax.text(v * 100 + 1, i, f"{v*100:.1f}%", va="center")
    plt.tight_layout()
    st.pyplot(fig)