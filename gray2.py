import cv2
import numpy as np

def canny(image):
    ## converting copied file to gray
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ## Applying gaussian blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    ## Gradient image
    canny = cv2.Canny(blur, 50, 150)
    return canny

## function for region of interest
def roi(image):
    height = image.shape[0] #array for height
    polygons = np.array([
    [(200, height), (1100,height), (550,250)]
    ]) ##array to specify the coordinates of the triangle
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask) #bitwise function, refer notes
    return masked_image

image = cv2.imread('test_image.jpg')
## making a copy of image
lane_image = np.copy(image)

## Display image from function
canny = canny(lane_image)
cropped_image = roi(canny)
cv2.imshow('result', roi(cropped_image))
cv2.waitKey(0)
cv2.destroyAllWindows()
