# app.py
import streamlit as st
from src.preprocess import get_processed_image  # Importing logic from Phase 1 file

# 1. Page Configuration
st.set_page_config(page_title="Braille AI Translator", layout="wide")
st.title("Braille to Normal Language Converter")
st.subheader("Phase 1: Image Upload & Preprocessing")

# 2. Sidebar/Instructions
st.sidebar.info("Upload a clear image of Braille dots on paper.")

# 3. File Uploader Widget
uploaded_file = st.file_uploader("Choose a Braille image file", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # 4. Process the image using our src file
    # We pass the 'read()' data to our function
    original_img, processed_img = get_processed_image(uploaded_file.read())

    # 5. Display the results in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 1. Original Upload")
        st.image(original_img, channels="BGR", use_container_width=True)
        
    with col2:
        st.write("### 2. AI-Ready Processed Image")
        st.image(processed_img, caption="Dots isolated for Deep Learning", use_container_width=True)

    st.success("Image successfully preprocessed! Ready for Phase 2 (CNN Prediction).")