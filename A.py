import cv2
import numpy as np


img = cv2.imread('Photos/ISIC_0086632.jpg')

def rescaleFrame(frame, scale=0.1):
    #for images and vids
    width = int(frame.shape[1] * scale)
    height = int( frame.shape[0]*scale)

    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation= cv2.INTER_AREA)

resized_image = rescaleFrame(img)

cv2.imshow('orig', resized_image)

gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

assert img is not None, "file could not be read, check with os.path.exists()"
# global thresholding

# Gamma correction function
def gamma_correction(image, gamma=2):
    # Apply gamma correction
    corrected_image = np.uint8(((image / 255.0) ** gamma) * 255)
    return corrected_image

# Apply gamma correction (adjust gamma value as needed)
gamma_value = 2 # Adjust gamma value here
corrected_img = gamma_correction(gray, gamma_value)

ret1,th1 = cv2.threshold(corrected_img,10,255,cv2.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv2.threshold(corrected_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(corrected_img,(301,301),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# plot all the images and their histograms


contours, hierarchy = cv2.findContours(image=th3, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

# create an empty mask
mask = np.zeros(corrected_img.shape[:2], dtype=np.uint8)

# loop through the contours
for i, cnt in enumerate(contours):
    # if the contour has no other contours inside of it
    if hierarchy[0][i][2] == -1:
        # if the size of the contour is greater than a threshold
        if cv2.contourArea(cnt) > 100000:
            cv2.drawContours(mask, [cnt], 0, (255), -1)
        # display result

cv2.imshow("Img", corrected_img)


#Draw the contours on the result image
cv.drawContours(img, contours, -1, (0, 255, 0), 2)"""

"""resized_image4 = rescaleFrame4(mask)"""
resized_image1 = rescaleFrame(mask)

cv2.imshow('resized', resized_image1)

# Find contours
contours, hierarchy = cv2.findContours(resized_image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
cv2.drawContours(resized_image, contours, -1, (255, 255, 0), 2)

# Calculate asymmetry for each contour
for contour in contours:
    # Compute the area of the contour
    area = cv2.contourArea(contour)

    # Find the centroid of the contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])

        # Print the centroid coordinates
        print("Centroid Coordinates:", centroid_x, centroid_y)


        # Draw a dot at the centroid on the image
        cv2.circle(resized_image, (centroid_x, centroid_y), 5, (0, 255, 0), -1)  # Green dot

        # Define the boundaries of the ROI
        x, y, w, h = cv2.boundingRect(contour)
        roi = corrected_img[y:y+h, x:x+w]



        # Compute the centroid of the ROI
        roi_centroid_x = centroid_x - x
        roi_centroid_y = centroid_y - y

   # Draw a dot at the centroid on the image
        cv2.circle(resized_image, (x + roi_centroid_x, y+3), 5, (0, 255, 0), -1)  # Green dot
        print("Coordinates Of Top Point:", x + roi_centroid_x, y+3)

        # Draw a dot at the centroid on the image
        cv2.circle(resized_image, (x + roi_centroid_x, y + h-8), 5, (0, 255, 0), -1)  # Green dot
        print("Coordinates Of Bottom Point :", x + roi_centroid_x, y + h-8)

        # Draw a dot at the centroid on the image
        cv2.circle(resized_image, (x+1, y + roi_centroid_y), 5, (0, 255, 0), -1)  # Green dot
        print("Coordinates Of Left Point :", x+1, y + roi_centroid_y)

        # Draw a dot at the centroid on the image
        cv2.circle(resized_image, (x -9 + w, y + roi_centroid_y), 5, (0, 255, 0), -1)  # Green dot
        print("Coordinates Of Right Point :", x -9+ w, y + roi_centroid_y)

        Radius_Top = centroid_y - (y+3)
        print("Radius Top: ", Radius_Top)

        Radius_Bottom = (y + h-8) - centroid_y
        print("Radius Bottom: ", Radius_Bottom)

        Radius_Left = centroid_x - (x+1)
        print("Radius Left: ", Radius_Left)

        Radius_Right = (x-9+ w) - centroid_x
        print("Radius Right: ", Radius_Right)

        Difference1 = np.abs(Radius_Top - Radius_Bottom)
        print("Difference between top and bottom radius: ", Difference1)
        Difference2 = np.abs(Radius_Left - Radius_Right)
        print("Difference between right and left radius: ", Difference2)
        Total = Difference1 + Difference2
        print("Total difference: ", Total)

# Display the image with contours and centroid dots
cv2.imshow('Contours with Centroid', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()






