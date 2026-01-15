import streamlit as st
import tensorflow as tf
import numpy as np
import os
from src.preprocess import get_processed_image, get_character_segments
from deep_translator import GoogleTranslator

st.set_page_config(page_title="Braille AI", layout="wide")

# UI Styling
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; }
    .logo-text { color: #007BFF; font-weight: bold; font-size: 30px; margin-bottom: 10px; }
    .result-box { font-size: 40px !important; font-weight: bold; color: #007BFF; background: #FFF; padding: 20px; border: 1px solid #DDD; border-radius: 10px; min-height: 150px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="logo-text">Braille AI</div>', unsafe_allow_html=True)

# Settings
target_lang = st.sidebar.selectbox("Target Language", ["English", "Malayalam", "Tamil", "Telugu"])
lang_codes = {"Malayalam": "ml", "Tamil": "ta", "Telugu": "te", "English": "en"}

# Load AI Model
@st.cache_resource
def load_braille_ai():
    path = "models/braille_model.h5" 
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

model = load_braille_ai()
class_names = list("abcdefghijklmnopqrstuvwxyz")

uploaded_file = st.file_uploader("Upload Braille Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    original, thresh_normal, thresh_inv = get_processed_image(uploaded_file.read())
    
    col_left, col_right = st.columns(2)

    with col_left:
        with st.container(border=True):
            st.write("### 📷 Frame 1: Braille Script")
            st.image(original, use_container_width=True)

    with col_right:
        with st.container(border=True):
            st.write(f"### 📜 Frame 2: Translation ({target_lang})")
            
            if model is None:
                st.error("Model file 'braille_model.h5' not found.")
            else:
                try:
                    # Segment characters
                    cells_data = get_character_segments(thresh_inv, thresh_normal)
                    
                    if len(cells_data) > 0:
                        # --- ADDED DEBUG PREVIEW ---
                        st.write("### AI Input Preview (Should be White dots on Black)")
                        # cells_data is a list of (image, is_space), so we take c[0]
                        st.image([c[0] for c in cells_data], width=50) 
                        # ---------------------------

                        eng_text = ""
                        for cell_img, has_space in cells_data:
                            # Rescale and Reshape
                            inp = cell_img.astype(np.float32) / 255.0
                            inp = inp.reshape(1, 28, 28, 1)
                            
                            # Prediction
                            pred = model.predict(inp, verbose=0)
                            eng_text += class_names[np.argmax(pred)]
                            if has_space: eng_text += " "
                        
                        final_eng = eng_text.upper()
                        
                        # Translate
                        if target_lang == "English":
                            final_output = final_eng
                        else:
                            final_output = GoogleTranslator(source='en', target=lang_codes[target_lang]).translate(final_eng)

                        st.markdown(f'<div class="result-box">{final_output}</div>', unsafe_allow_html=True)
                        if target_lang != "English":
                            st.caption(f"English: {final_eng}")
                    else:
                        st.warning("No Braille cells detected.")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
else:
    st.info("Please upload an image to view results.")