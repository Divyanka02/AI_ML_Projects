import cv2

directory1 = r'F:/Python_classes/PYTHON AI/OpenCV/' # Source Folder
directory2 = r'F:/Python_classes/PYTHON AI/OpenCV/' # Destination Folder

image = cv2.imread(directory1+'road.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite(directory2+'image1_gray.png', gray)

cv2.waitKey(0)
cv2.destroyAllWindows()