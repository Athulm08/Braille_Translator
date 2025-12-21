import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
from src.preprocess import get_processed_image

# 1. Setup Page
st.set_page_config(page_title="Braille AI Translator", layout="wide")
st.title("Braille to English AI Translator")

# 2. Define Character Mapping (A-Z)
class_names = list("abcdefghijklmnopqrstuvwxyz")

# 3. Load the Trained Model (Phase 2 result)
@st.cache_resource
def load_braille_model():
    model_path = 'models/braille_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_braille_model()

# 4. Interface
uploaded_file = st.file_uploader("Upload a Braille image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # --- PHASE 1: Preprocessing ---
    # We get the 'processed' (binary) image from your src file
    original_img, processed_img = get_processed_image(uploaded_file.read())
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original_img, channels="BGR", use_container_width=True)
    with col2:
        st.subheader("AI Processed Image")
        st.image(processed_img, caption="Dots isolated", use_container_width=True)

    # --- PHASE 3: Prediction ---
    if model is not None:
        st.write("---")
        st.subheader("AI Translation Result")
        
        # A. Resize to 28x28 (This is what your AI was trained on)
        resized_img = cv2.resize(processed_img, (28, 28))
        
        # B. Normalize (0-1)
        normalized_img = resized_img / 255.0
        
        # C. Reshape for the AI (Batch size, Height, Width, Channels)
        final_input = np.reshape(normalized_img, (1, 28, 28, 1))
        
        # D. Predict
        with st.spinner('AI is thinking...'):
            prediction = model.predict(final_input)
            result_index = np.argmax(prediction) # Get the highest score
            confidence = np.max(prediction) * 100
            translated_letter = class_names[result_index]

        # E. Display Result
        st.success(f"Translated Letter: **{translated_letter.upper()}**")
        st.info(f"AI Confidence: {confidence:.2f}%")
    else:
        st.error("Error: 'braille_model.h5' not found in 'models' folder. Please run Phase 2 training first!")