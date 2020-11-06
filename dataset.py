import preprocess as pp
import numpy as np
import os
import cv2

def readFile(filepath, bin, verbose=False):
    input = []
    output = []

    dir = os.listdir(filepath)
    for i in dir:

        name_splitted = os.path.splitext(i)

        if name_splitted[1] == '.png':
            img = cv2.imread(filepath+i, 0)
            img = pp.binarization(img, bin)
            if verbose:
                cv2.imshow("Demo(Press 'ESC' to quit.)", img)
                print("Press 'ESC' to quit.")
                key = cv2.waitKey(0)
                if key == 27:
                    cv2.destroyAllWindows()
            else:
                input.append([img])

            # Label the output. Health:0, Parkinson:1.
            if "H" in name_splitted[0]:
                output.append([0])
            elif "P" in name_splitted[0]:
                output.append([1])
            elif name_splitted[0] == "sample"
            else:
                print("Unlabelled sample. Fix me please: "+filepath+i)

    return


def makeDataset(filepath):
    """
    input: black 0, none 1, white 2, next_move 3
    output: black -1, none 0, white 1
    :param filepath: string with extensions if there are
    :return:
    """
    str = ""

    input0 = []
    input1 = []
    output0 = []
    output1 =[]


    input0 = np.array(input0,dtype = np.float32)
    input1 = np.array(input1, dtype = np.float32)
    output0 = np.array(output0,dtype = np.int64)
    output1 = np.array(output1, dtype=np.int64)

    return input0,input1,output0,output1


if __name__ == "__main__":
    bin = np.arange(256)
    print(readFile("./", bin, verbose=True))