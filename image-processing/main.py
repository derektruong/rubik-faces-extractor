import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./dataset/Frame_Down.jpg')
dst = cv.bilateralFilter(img,9,75,75)
dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
hsv_img = cv.cvtColor(dst, cv.COLOR_BGR2HSV)   # HSV image

LOWERS = {
	'yellow': np.array([20, 100, 100],np.uint8),
	'yellow': np.array([20, 100, 100],np.uint8),
	'yellow': np.array([20, 100, 100],np.uint8),
	'yellow': np.array([20, 100, 100],np.uint8),
}

COLOR_MIN = np.array([20, 100, 100],np.uint8)       # HSV color code lower and upper bounds
COLOR_MAX = np.array([30, 255, 255],np.uint8)       # color yellow 

frame_threshed = cv.inRange(hsv_img, COLOR_MIN, COLOR_MAX)     # Thresholding image
imgray = frame_threshed
ret,thresh = cv.threshold(frame_threshed,127,255,0)
contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
# print type(contours)
for cnt in contours:
    x,y,w,h = cv.boundingRect(cnt)
    print(x,y)
    cv.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)
cv.imshow("Show",dst)
cv.imwrite("extracted.jpg", dst)
cv.waitKey()
cv.destroyAllWindows()

# plt.subplot(121),plt.imshow(img)
# plt.subplot(122),plt.imshow(dst)
# plt.show()
