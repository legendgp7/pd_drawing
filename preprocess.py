#!/usr/bin/env python
import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys

def show(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def denoise(img, neighbor=2):
    """
    foreground pixel: 1 ; background pixel: 0
    """

    imgp = np.pad(img, 1, 'constant', constant_values=0)

    leny = img.shape[0]
    lenx = img.shape[1]
    for i in range(lenx):
        for j in range(leny):
            if img[j,i] == 0:
                continue
            elif img[j,i] == 1:
                n = np.sum(imgp[j:j+3,i:i+3])
                img[j,i] = 1 if n > neighbor else 0
            else:
                print("Warning: the foreground and background pixel should be 1 and 0 resepctively!")
                break

    return img


def binarization(img, bin, c=12, verbose=False, detail=True, denoise_on=True, save_flag=False):
    imgf = img.flatten()
    his = np.histogram(imgf, bin)[0]
    d1 = his[1:255] - his[0:254]
    d2 = d1[1:254] - d1[0:253]

    idx_peak = np.argmax(his)
    th = np.argmax(d2[0:idx_peak]) - c

    if verbose:
        plt.hist(imgf,bin)
        plt.title("histogram")
        if save_flag:
            plt.savefig('Demo_origin_histogram.png')
        else:
            plt.show()

    if detail:
        binary = cv2.threshold(img, th, 1, cv2.THRESH_BINARY_INV)[1]

        if denoise_on:
            binary = denoise(binary)

        img_filtered = binary * img
        min_val = img.min()
        max_val = img.max()
        return (img_filtered - min_val) / (max_val - min_val)
    else:
        return cv2.threshold(img, th, 255, cv2.THRESH_BINARY)[1]


if __name__ == "__main__":

    save_flag = "-s" in sys.argv

    img = cv2.imread("sample.png",0)
    bin = np.arange(256)
    img2 = binarization(img,bin,verbose=True, save_flag=save_flag)

    if save_flag:
        cv2.imwrite('Demo_binarization.png',img2*255)
    else:
        cv2.imshow("Demo(Press 'ESC' to quit.)",img2)
        print("Press 'ESC' to quit.")
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
