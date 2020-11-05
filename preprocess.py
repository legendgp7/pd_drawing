import cv2
from matplotlib import pyplot as plt
import numpy as np


def binarization(img, bin, c=30, verbose=False, detail=True):
    imgf = img.flatten()
    his = np.histogram(imgf, bin)[0]
    diff = his[1:255] - his[0:254]
    th = np.argmax(diff) - c
    if verbose:
        plt.hist(imgf,bin)
        plt.title("histogram")
        plt.show()

    if detail:
        binary = cv2.threshold(img, th, 1, cv2.THRESH_BINARY_INV)[1]
        img_filtered = binary * img
        min_val = img.min()
        max_val = img.max()
        return (img_filtered - min_val) / (max_val - min_val)
    else:
        return cv2.threshold(img, th, 255, cv2.THRESH_BINARY)[1]


img = cv2.imread("sample.png",0)
bin = np.arange(256)
img2 = binarization(img,bin,verbose=True)
cv2.imshow("Demo(Press 'ESC' to quit.)",img2)
print("Press 'ESC' to quit.")
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()