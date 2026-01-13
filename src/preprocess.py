#braile_translator/src/ preprocess.py
import cv2
import numpy as np

def get_processed_image(image_bytes):
    # 1. Convert bytes to OpenCV image
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 2. Basic Cleaning
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Create Normal Threshold (Black dots on White)
    thresh_normal = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 4. Create Inverted Threshold (White dots on Black)
    # THIS IS THE 3rd VALUE THE APP IS LOOKING FOR
    thresh_inv = cv2.bitwise_not(thresh_normal)
    
    return img, thresh_normal, thresh_inv 

def get_character_segments(binary_image):
    # Bridge the dots into one character block
    kernel = np.ones((25, 15), np.uint8)
    dilated = cv2.dilate(binary_image, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small noise and sort Reading Order
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50]
    boxes = sorted(boxes, key=lambda b: (b[1] // 50, b[0])) 

    cells = []
    for (x, y, w, h) in boxes:
        roi = binary_image[y:y+h, x:x+w]
        
        # Create square padding to prevent dot distortion
        size = max(w, h) + 10
        square = np.zeros((size, size), dtype="uint8")
        dx, dy = (size - w) // 2, (size - h) // 2
        square[dy:dy+h, dx:dx+w] = roi
        
        # Resize to 28x28
        cells.append(cv2.resize(square, (28, 28)))
        
    return cells# braile_translator/src/preprocess.py
import cv2
import numpy as np


def get_processed_image(image_bytes):
    # 1. Convert bytes to OpenCV image
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # 2. Basic cleaning
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Normal threshold (black dots on white)
    thresh_normal = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    # 4. Inverted threshold (white dots on black) – usually better for the model
    thresh_inv = cv2.bitwise_not(thresh_normal)

    return img, thresh_normal, thresh_inv


def get_character_segments(binary_image):
    # Connect dots into one character block
    kernel = np.ones((25, 15), np.uint8)
    dilated = cv2.dilate(binary_image, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter noise and sort in reading order
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50]
    boxes = sorted(boxes, key=lambda b: (b[1] // 50, b[0]))

    cells = []
    for (x, y, w, h) in boxes:
        roi = binary_image[y : y + h, x : x + w]

        # Square padding
        size = max(w, h) + 10
        square = np.zeros((size, size), dtype="uint8")
        dx, dy = (size - w) // 2, (size - h) // 2
        square[dy : dy + h, dx : dx + w] = roi

        # Resize to 28x28
        cells.append(cv2.resize(square, (28, 28)))

    return cells
