import streamlit as st
from PIL import Image
import cv2
import numpy as np
from predict import predict_image

st.set_page_config(page_title="Corrosion Detection App", layout="centered")
st.title("🧪 Corrosion Detection using AI")
st.markdown("Upload an image or use webcam to detect corrosion.")
st.sidebar.title("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

option = st.radio("Choose Input Source:", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_container_width=True)
        result = predict_image(img)
        st.success(f"🔍 Result: {result}")
else:
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([], use_container_width=True)
    camera = cv2.VideoCapture(0)
    while run:
        _, frame = camera.read()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(img_rgb)
        img_pil = Image.fromarray(img_rgb)
        result = predict_image(img_pil)
        st.write(f" Result: {result}")
    else:
        st.write('Stopped')
