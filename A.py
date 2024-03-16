import cv2
import numpy as np

# Read the image
img = cv2.imread('Photos/ISIC_0086632.jpg')

# Function to rescale the frame
def rescaleFrame(frame, scale=0.1):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Rescale the image
resized_image = rescaleFrame(img)

# Show the original image
cv2.imshow('Original Image', resized_image)

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gamma correction function
def gamma_correction(image, gamma=2):
    corrected_image = np.uint8(((image / 255.0) ** gamma) * 255)
    return corrected_image

# Apply gamma correction
gamma_value = 2
corrected_img = gamma_correction(gray, gamma_value)

# Thresholding
ret1, th1 = cv2.threshold(corrected_img, 10, 255, cv2.THRESH_BINARY)
ret2, th2 = cv2.threshold(corrected_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(corrected_img, (301, 301), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find contours
contours, hierarchy = cv2.findContours(image=th3, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

# Create an empty mask
mask = np.zeros(corrected_img.shape[:2], dtype=np.uint8)

# Loop through the contours
for i, cnt in enumerate(contours):
    if hierarchy[0][i][2] == -1:
        if cv2.contourArea(cnt) > 100000:
            cv2.drawContours(mask, [cnt], 0, (255), -1)

# Show the corrected image
cv2.imshow("Corrected Image", corrected_img)

# Rescale the mask
resized_mask = rescaleFrame(mask)
cv2.imshow('Resized Mask', resized_mask)

# Find contours on the resized mask
contours, hierarchy = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
cv2.drawContours(resized_image, contours, -1, (255, 255, 0), 2)

# Calculate asymmetry for each contour
for contour in contours:
    area = cv2.contourArea(contour)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])

        print("Centroid Coordinates:", centroid_x, centroid_y)

        x, y, w, h = cv2.boundingRect(contour)

        roi_centroid_x = centroid_x - x
        roi_centroid_y = centroid_y - y

        cv2.circle(resized_image, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
        cv2.circle(resized_image, (x + roi_centroid_x, y + 3), 5, (0, 255, 0), -1)
        cv2.circle(resized_image, (x + roi_centroid_x, y + h - 8), 5, (0, 255, 0), -1)
        cv2.circle(resized_image, (x + 1, y + roi_centroid_y), 5, (0, 255, 0), -1)
        cv2.circle(resized_image, (x - 9 + w, y + roi_centroid_y), 5, (0, 255, 0), -1)

        Radius_Top = centroid_y - (y + 3)
        Radius_Bottom = (y + h - 8) - centroid_y
        Radius_Left = centroid_x - (x + 1)
        Radius_Right = (x - 9 + w) - centroid_x

        Difference1 = np.abs(Radius_Top - Radius_Bottom)
        Difference2 = np.abs(Radius_Left - Radius_Right)
        Total = Difference1 + Difference2

        print("Coordinates Of Top Point:", x + roi_centroid_x, y + 3)
        print("Coordinates Of Bottom Point :", x + roi_centroid_x, y + h - 8)
        print("Coordinates Of Left Point :", x + 1, y + roi_centroid_y)
        print("Coordinates Of Right Point :", x - 9 + w, y + roi_centroid_y)
        print("Radius Top: ", Radius_Top)
        print("Radius Bottom: ", Radius_Bottom)
        print("Radius Left: ", Radius_Left)
        print("Radius Right: ", Radius_Right)
        print("Difference between top and bottom radius: ", Difference1)
        print("Difference between right and left radius: ", Difference2)
        print("Total difference: ", Total)

# Show the image with contours and centroid dots
cv2.imshow('Contours with Centroid', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()





