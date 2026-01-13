# braile_translator/app.py

import os
import numpy as np
import tensorflow as tf
import streamlit as st

from src.preprocess import get_processed_image, get_character_segments


# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Braille → English Translator", layout="wide")
st.title("Braille Image to English Text Translator")

st.sidebar.header("Options")
fix_inversion = st.sidebar.checkbox(
    "Use normal threshold (check if output looks wrong)",
    value=False,
)


# ---------- LOAD MODEL ----------
def load_model():
    model_path = "models/braille_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None


model = load_model()
class_names = list("abcdefghijklmnopqrstuvwxyz")  # must match training labels


# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader(
    "Upload a Braille image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Please upload a Braille image to start.")
elif model is None:
    st.error("Model file not found at 'models/braille_model.h5'. Train or copy it first.")
else:
    # ---------- IMAGE PREPROCESSING ----------
    # 1. Get original and binary versions
    original, thresh_normal, thresh_inv = get_processed_image(
        uploaded_file.read()
    )

    # 2. Choose which binary image to send to the model
    #    If output looks inverted/wrong, toggle the sidebar checkbox.
    ai_input_binary = thresh_normal if fix_inversion else thresh_inv

    # 3. Segment into individual Braille character cells
    cells = get_character_segments(ai_input_binary)

    # ---------- DISPLAY IMAGES ----------
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original Image", use_container_width=True)
    with col2:
        st.image(
            ai_input_binary,
            caption="Binary Image Used for Prediction",
            use_container_width=True,
        )

    if len(cells) == 0:
        st.warning("No Braille dots detected. Try a clearer image.")
    else:
        st.write("---")
        st.subheader("Detected Character Cells")
        st.image(cells, width=60)

        # ---------- PREDICTION LOOP ----------
        english_text = ""
        for cell in cells:
            # normalize to 0–1 and reshape to (1, 28, 28, 1)
            inp = cell.astype("float32") / 255.0
            inp = inp.reshape(1, 28, 28, 1)
            prediction = model.predict(inp, verbose=0)
            english_text += class_names[np.argmax(prediction)]

        english_text = english_text.upper()

        # ---------- OUTPUT ----------
        st.subheader("Translated English Text")
        st.success(english_text)
