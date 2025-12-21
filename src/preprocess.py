# src/preprocess.py
import cv2
import numpy as np

def get_processed_image(image_bytes):
    """
    Takes raw image bytes, cleans the image, and returns 
    the original and the binary (processed) version.
    """
    # 1. Convert bytes to an OpenCV image format
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1) # Color image
    
    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Blur (To remove paper grain/noise)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. Adaptive Thresholding (Turns dots white, paper black)
    # This handles different lighting conditions
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return img, thresh

def segment_dots(binary_image):
    """
    Future Phase: This will be used to crop the image into 
    individual Braille cells.
    """
    pass