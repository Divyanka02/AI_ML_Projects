import cv2
import numpy as np

def cartoonize(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 9, 300, 300)
    
    # Detect edges in the image using adaptive thresholding
    edges = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 9, 2)
    
    # Apply a median blur to the thresholded image
    edges = cv2.medianBlur(edges, 5)
    
    # Create a color image using bitwise_and operator between image and edges
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    
    return cartoon

# Example usage
image_path = "F:\Python_classes\Divyanka_photo.jpg"
cartoon_image = cartoonize(image_path)
cv2.imshow("Cartoonized Image", cartoon_image)
cv2.waitKey(0)
cv2.destroyAllWindows()