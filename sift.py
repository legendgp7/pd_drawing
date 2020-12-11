import xgboost as xgb

from sklearn.decomposition import PCA
import dataset as ds
import numpy as np
import sys
import aug
import time
import cv2
from matplotlib import pyplot as plt
import kmeans as km


best = 0
for k in range(5,6):
    for l in range(20,21):
        if k>l:
            continue

        correct_h = 0
        correct_p = 0
        kp_list_h = []
        kp_list_p = []
        coef = 0.55*l/k
        orb = cv2.ORB_create(nfeatures=l)

        for i in range(1, 41):
            img = cv2.imread('processed_dataset/H%d.png' % i, 0)
            kp = orb.detect(img, None)
            kp_list_h.append(cv2.KeyPoint_convert(kp))

        for i in range(52, 92):
            img = cv2.imread('processed_dataset/P%d.png' % i, 0)
            kp = orb.detect(img, None)
            kp_list_p.append(cv2.KeyPoint_convert(kp))

        kp_list_h = np.concatenate(kp_list_h)
        kp_list_p = np.concatenate(kp_list_p)

        cp, cluster = km.KMeans(kp_list_h, k)
        cp2, cluster2 = km.KMeans(kp_list_p, k)

        kp_list_h = cp
        kp_list_p = cp2

        for i in range(41, 52):
            img = cv2.imread('processed_dataset/H%d.png' % i, 0)
            kp = orb.detect(img, None)
            points = cv2.KeyPoint_convert(kp)

            n = points.shape[0]
            dist_h = 0
            dist_p = 0
            nh = 0
            np0 = 0

            for j in range(n):
                dist_h = np.sqrt(np.sum(np.square(kp_list_h - points[j]), axis=1)).min()
                dist_p = np.sqrt(np.sum(np.square(kp_list_p - points[j]), axis=1)).min()
                if dist_h <= dist_p:
                    nh += 1
                else:
                    np0 += 1

            if nh*coef>=np0:
                correct_h += 1








        for i in range(93, 103):
            img = cv2.imread('processed_dataset/P%d.png' % i, 0)
            kp = orb.detect(img, None)
            points = cv2.KeyPoint_convert(kp)
            n = points.shape[0]
            dist_h = 0
            dist_p = 0

            np0 = 0
            nh = 0
            for j in range(n):
                dist_h = np.sqrt(np.sum(np.square(kp_list_h - points[j]), axis=1)).min()
                dist_p = np.sqrt(np.sum(np.square(kp_list_p - points[j]), axis=1)).min()
                if dist_h <= dist_p:
                    nh += 1
                else:
                    np0 += 1

            if nh <= np0*coef:
                correct_p += 1



        #print(correct_h,correct_p)
        acc = correct_h+correct_p
        acc = acc / 22


        if acc>best:
            best = acc
        print("center: %d, feature: %d, acc = %.4f" % (k, l, acc))
        print("best: %.4f"%best)
