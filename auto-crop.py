import cv2
 
import numpy as np
 
f_path = ""

# Load the image
image = cv2.imread(f_path)

cv2.imshow("Original Image", image)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply image thresholding to create a binary mask
_, thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply morphological operations to enhance the mask
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)

# Find contours in the mask
contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate over the contours and find the contour with the largest area
largest_contour = max(contours, key=cv2.contourArea)

# Get the bounding rectangle of the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Crop the fruit from the original image using the bounding rectangle
fruit_crop = image[y:y+h, x:x+w]

# Display the cropped fruit image
cv2.imshow("Cropped Fruit", fruit_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()