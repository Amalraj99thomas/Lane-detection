import cv2
import numpy as np

image = cv2.imread('test_image.jpg')
## making a copy of image
lane_image = np.copy(image)
## converting copied file to gray
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
## Applying gaussian blur
blur = cv2.GaussianBlur(gray, (5,5), 0)
## Gradient image
canny = cv2.Canny(blur, 50, 150)
## Display grey image
cv2.imshow('result', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
