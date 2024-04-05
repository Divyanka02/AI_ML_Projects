
# pip install opencv-python
import cv2
import numpy as np

def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum
    # The bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # The top-right point will have the smallest difference
    # The bottom-left point will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts):
    # Obtain a consistent order of the points and unpack them
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left
    # or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute the height of the new image, which will be the maximum distance between the top-right and bottom-right
    # or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Construct the set of destination points to obtain a "birds eye view" (i.e., top-down view) of the image
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    # Return the warped image
    return warped

def doc_scanner(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection
    edged = cv2.Canny(blurred, 75, 200)
    
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the contour with the maximum area
    contour = max(contours, key=cv2.contourArea)
    
    # Get the perimeter of the contour
    perimeter = cv2.arcLength(contour, True)
    
    # Approximate the contour by a polygon
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # If the contour has four points, it is likely a document
    if len(approx) == 4:
        # Apply perspective transform to obtain a top-down view of the document
        warped = four_point_transform(image, approx.reshape(4, 2))
        
        # Return the scanned document
        return warped
    
    else:
        print("Document not detected.")
        return None

# Main function
def main():
    # Path to the input image
    image_path = "F:\Python_classes\image2.png"
    
    # Perform document scanning
    scanned_document = doc_scanner(image_path)
    
    # Display the original and scanned document
    if scanned_document is not None:
        cv2.imshow("Original Document", cv2.imread(image_path))
        cv2.imshow("Scanned Document", scanned_document)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main()