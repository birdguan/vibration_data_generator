from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

def prepro():
    file = loadmat('waiquan716_256_3.mat')
    file_keys = file.keys()
    data = []
    for key in file_keys:
        if 'data' in key:
            data.append(file[key])
    X_train = data[0] # 716 * 196608
    print(X_train.shape)

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        image_arr = X_train[i].reshape((3, 256, 256))
        # print(image_arr.shape)(3, 256, 256)
        image_arr = image_arr.transpose((2, 1, 0))  # 通道的交换 256*256*3
        # print(image_arr.shape)(256, 256, 3)
        plt.imshow(image_arr)
    plt.show()
    X_train = X_train.reshape(-1, 256, 256, 3)
    return X_train


if __name__ == "__main__":
    X_train = prepro()
    print(X_train.shape)