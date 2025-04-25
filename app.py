import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

os.environ["STREAMLIT_WATCH_USE_POLLING"] = "true"

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom',
                          path='../yolov5/runs/train/fish_detector7/weights/best.pt',
                          force_reload=True)

model = load_model()

# Sidebar UI
st.sidebar.title("⚙️ Options")
input_type = st.sidebar.radio("Choose Input Type", ["Image"])

# Main UI
st.title("🐟 Fish Detection & Analysis App")
st.markdown("Upload an image to detect and count fish using YOLOv5.")

# Handle file upload based on input type
if input_type == "Image":
    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            st.subheader("📷 Uploaded Image")
            st.image(image, caption="Original Image", use_container_width=True)

            results = model(image_np)
            detections = results.xyxy[0]
            fish_count = len(detections)
            st.success(f"✅ Fish Detected: {fish_count}")

            with st.expander("🔍 Detection Details"):
                for i, det in enumerate(detections):
                    x1, y1, x2, y2, conf, cls = det
                    st.write(f"Fish {i+1}: Confidence `{conf:.2f}` - Coordinates ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")

            results.render()
            st.image(np.squeeze(results.ims[0]), caption="📦 Detected Fish", use_container_width=True)
        else:
            st.error("❌ Please upload a valid image file.")
