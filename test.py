import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('processed_dataset/P78.png',0)          # queryImage


print(img.shape)
# Initiate ORB detector
orb = cv2.ORB_create(nfeatures=160)
# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)



# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()
"""
orb = cv2.ORB_create()
# find the keypoints and descriptors with SIFT
kp1 = orb.detect(img1,None)
print(img1)
print(kp1)
# Initiate SIFT detector
sift = cv2.SIFT()
"""
