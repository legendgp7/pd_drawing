import preprocess as pp
import numpy as np
import os
import cv2
import configparser

def readFile(input, output, filepath, bin, n, verbose=False):

    dir = os.listdir(filepath)
    for i in dir:

        name_splitted = os.path.splitext(i)

        if name_splitted[1] == '.png':
            n["cnt"] += + 1

            if n["cnt"]%10 == 0:
                print("%d images loaded..."% n["cnt"])

            img = cv2.imread(filepath + "/" + i, 0)
            if img.shape[0] != 256 or img.shape[1] != 256:
                img = cv2.resize(img, (256,256))
            img = pp.binarization(img, bin)

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


def makeDataset(bin, type="s"):
    """
    :param type: spiral "s"; wave "w";
    :return: x training set
             y testing set
    """
    n = {"cnt": 0}
    x = []
    y = []

    config = configparser.ConfigParser()
    config.read('path.ini')
    abs_path = config["absolute"]["path"]
    if type == "w":
        print("Reading wave images...")
        readFile(x, y, abs_path + config["wave"]["healthytrain"], bin, n)
        readFile(x, y, abs_path + config["wave"]["healthytest"], bin, n)
        readFile(x, y, abs_path + config["wave"]["parkinsontrain"], bin, n)
        readFile(x, y, abs_path + config["wave"]["parkinsontest"], bin, n)
    elif type == "t":
        readFile(x, y, abs_path + config["spiral"]["healthytest"], bin, n)
    else:
        print("Reading spiral images...")
        readFile(x, y, abs_path + config["spiral"]["healthytrain"], bin, n)
        readFile(x, y, abs_path + config["spiral"]["healthytest"], bin, n)
        readFile(x, y, abs_path + config["spiral"]["parkinsontrain"], bin, n)
        readFile(x, y, abs_path + config["spiral"]["parkinsontest"], bin, n)

    print("Dataset prepared (%d images)"%n["cnt"])
    x = np.array(x, dtype = np.float32)
    y = np.array(y, dtype = np.int64)

    return x, y


if __name__ == "__main__":
    bin = np.arange(256)
    x, y = makeDataset(bin)

    print("Input shape: ", x.shape)
    print("Output shape: ", y.shape)