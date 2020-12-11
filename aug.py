import copy
import numpy as np

def data_augment(x_train, y_train,size):
    num_train = x_train.shape[0]
    x = x_train.reshape((num_train, size, size))

    x_f = np.fliplr(x)
    x = np.concatenate((x, x_f), axis=0)

    y = np.concatenate((y_train, y_train), axis=0)

    x_out = copy.deepcopy(x)

    y_out = copy.deepcopy(y)

    for i in range(3):
        temp = np.rot90(x, k=i + 1, axes=(1, 2))
        x_out = np.concatenate((x_out, temp), axis=0)
        y_out = np.concatenate((y_out, y), axis=0)

    new_num = x_out.shape[0]
    x_out = x_out.reshape((new_num, size, size, 1))

    return x_out,  y_out
