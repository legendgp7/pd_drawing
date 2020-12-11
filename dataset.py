#!/usr/bin/env python
import preprocess as pp
import numpy as np
import os
import cv2
import configparser

def readFile(input, output, filepath, bin, n, size=256,verbose=False):

    dir = os.listdir(filepath)
    for i in dir:

        name_splitted = os.path.splitext(i)

        if name_splitted[1] == '.png':
            n["cnt"] += + 1

            if n["cnt"]%10 == 0:
                print("%d images loaded..."% n["cnt"])

            img = cv2.imread(filepath + "/" + i, 0)
            if img.shape[0] != size or img.shape[1] != size:
                img = cv2.resize(img, (size,size))

            if verbose:
                cv2.imshow("Demo(Press 'ESC' to quit.)", img)
                print("Press 'ESC' to quit.")
                key = cv2.waitKey(0)
                if key == 27:
                    cv2.destroyAllWindows()
            else:
                input.append(np.array([img]))

            # Label the output. Health:0, Parkinson:1.
            if "H" in name_splitted[0]:
                output.append(np.array([0]))
            elif "P" in name_splitted[0]:
                output.append(np.array([1]))
            elif name_splitted[0] == "sample":
                output.append(np.array([0]))
            else:
                print("Unlabelled sample. Fix me please: "+filepath+i)

    return

def saveFile(filepath, bin, n, size=256,verbose=False):

    dir = os.listdir(filepath)
    for i in dir:

        name_splitted = os.path.splitext(i)

        if name_splitted[1] == '.png':
            n["cnt"] += + 1

            if n["cnt"]%10 == 0:
                print("%d images loaded..."% n["cnt"])

            img = cv2.imread(filepath + "/" + i, 0)
            if img.shape[0] != size or img.shape[1] != size:
                img = cv2.resize(img, (size,size))
            img = pp.binarization(img, bin)

            if verbose:
                cv2.imshow("Demo(Press 'ESC' to quit.)", img)
                print("Press 'ESC' to quit.")
                key = cv2.waitKey(0)
                if key == 27:
                    cv2.destroyAllWindows()
            else:
                pass

            # Label the output. Health:0, Parkinson:1.
            if "H" in name_splitted[0]:
                cv2.imwrite('./processed_dataset/H%d.png'%n["cnt"], img * 255)

            elif "P" in name_splitted[0]:
                cv2.imwrite('./processed_dataset/P%d.png' % n["cnt"], img * 255)

            elif name_splitted[0] == "sample":
                cv2.imwrite('processed_sample.png' , img * 255)
            else:
                print("Unlabelled sample. Fix me please: "+filepath+i)

    return

def readDataset(bin, type="s"):
    """
    :param type: spiral "s"; wave "w";
    :return: x training set
             y testing set
    """
    n = {"cnt": 0}
    x = []
    y = []
    x2 = []
    y2 = []


    if type == "w":
        print("Reading wave images...")
        """readFile(x, y, abs_path + config["wave"]["healthytrain"], bin, n)
        readFile(x2, y2, abs_path + config["wave"]["healthytest"], bin, n)"""

    elif type == "t":
        readFile(x, y, "./processed_dataset", bin, n)
    else:
        print("Reading spiral images...")
        readFile(x, y, "./processed_dataset/", bin, n)

    x = np.array(x, dtype = np.float32)
    y = np.array(y, dtype = np.int64)
    x = x.transpose((0,2,3,1))

    print("Dataset prepared (%d images)\n" % n["cnt"])

    return x, y


def makeDataset(bin, type="s"):
    """
    :param type: spiral "s"; wave "w";
    :return: x training set
             y testing set
    """
    check_folder()
    n = {"cnt": 0}


    config = configparser.ConfigParser()
    config.read('path.ini')
    abs_path = config["absolute"]["path"]
    if type == "w":
        print("Reading wave images...")
        saveFile( abs_path + config["wave"]["healthytrain"], bin, n)
        saveFile(abs_path + config["wave"]["healthytest"], bin, n)
        saveFile(abs_path + config["wave"]["parkinsontrain"], bin, n)
        saveFile(abs_path + config["wave"]["parkinsontest"], bin, n)
    elif type == "t":
        saveFile(abs_path + config["spiral"]["healthytest"], bin, n)
    else:
        print("Reading spiral images...")
        saveFile(abs_path + config["spiral"]["healthytrain"], bin, n)
        saveFile(abs_path + config["spiral"]["healthytest"], bin, n)
        saveFile(abs_path + config["spiral"]["parkinsontrain"], bin, n)
        saveFile(abs_path + config["spiral"]["parkinsontest"], bin, n)


    print("Dataset prepared (%d images)\n" % n["cnt"])

    return

def check_folder():
    """
    Initialize the folder for the purpose of storing the processed images.
    """
    if os.path.isdir('./processed_dataset')==False:
        print("** Initializing the processed dataset.")
        os.mkdir('processed_dataset')
        return False
    else:
        return True

if __name__ == "__main__":
    bin = np.arange(256)
    makeDataset(bin, type="s")

    """x,y = readDataset(bin,type="s")
    print("Input shape (Number of samples, Y_size, X_size,  Channel): ", x.shape)
    print("Output shape (Number of samples, Output dimension): ", y.shape)"""