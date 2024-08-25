import cv2
import pytesseract
import numpy as np
print(pytesseract.get_tesseract_version())
image = cv2.imread('D:\\programming\\python\\Screenshot 2024-08-25 194825.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Applying a Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Sharpening using the Laplacian operator
sharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


# Detecting edges to find the rotation angle
edges = cv2.Canny(sharp, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# Finding the angle
if lines is not None:
    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90
        angles.append(angle)
    
    median_angle = np.median(angles)
    print(f"Detected angle: {median_angle}")

    # Rotating the image to correct the orientation
    (h, w) = sharp.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(sharp, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
else:
    rotated = sharp
# Binarize the image
_, binary = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# Apply OCR
text = pytesseract.image_to_string(binary, lang='eng')
print("Extracted Text:")
print(text)
# import cv2
# import pytesseract

# # Load an image
# image = cv2.imread('D:\\programming\\python\\Screenshot 2024-08-25 194825.png')

# # Use pytesseract to extract text
# text = pytesseract.image_to_string(image)

# print(text)
