import cv2
import numpy as np

def get_processed_image(image_bytes):
    """
    Returns original image, normal threshold (Black dots), 
    and inverted threshold (White dots).
    """
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 1. Normal: White Background, Black Dots
    thresh_normal = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 2. Inverted: Black Background, White Dots (Used for finding dot clusters)
    thresh_inv = cv2.bitwise_not(thresh_normal)
    
    return img, thresh_normal, thresh_inv

def get_character_segments(binary_inv, thresh_normal):
    """
    Segments the image into individual 28x28 characters.
    Inverts the dots so they are White on Black for the AI.
    """
    # Group dots into character blocks
    kernel = np.ones((25, 15), np.uint8)
    dilated = cv2.dilate(binary_inv, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter noise and sort Left-to-Right
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50]
    boxes = sorted(boxes, key=lambda b: (b[1] // 60, b[0])) 

    cells = []
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        
        # --- THE FIX STARTS HERE ---
        # 1. Crop from the image that has Black dots
        roi = thresh_normal[y:y+h, x:x+w]
        
        # 2. INVERT: Change Black dots to White, Background to Black
        # This matches what the AI expects and stops the "WWWW" result.
        roi = cv2.bitwise_not(roi) 
        
        # 3. Create a Square Canvas with Black background
        size = max(w, h) + 12
        square = np.zeros((size, size), dtype="uint8") 
        
        # 4. Center the white dots in the black square
        dx, dy = (size - w) // 2, (size - h) // 2
        square[dy:dy+h, dx:dx+w] = roi
        
        # 5. Resize to 28x28 (Final AI shape)
        cell_final = cv2.resize(square, (28, 28))
        # --- THE FIX ENDS HERE ---
        
        # Space detection logic
        is_space = False
        if i < len(boxes) - 1:
            next_x = boxes[i+1][0]
            current_x_end = x + w
            if (next_x - current_x_end) > (w * 0.8): 
                is_space = True
            
        cells.append((cell_final, is_space))
        
    return cells