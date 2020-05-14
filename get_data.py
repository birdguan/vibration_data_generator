from scipy.io import loadmat
import cv2

def parse_imgs():
    file = loadmat('waiquan716_256_3.mat')
    file_keys = file.keys()
    data = []
    for key in file_keys:
        if  'data' in key:
            data.append(file[key])

    datas = data[0]
    for i in range(len(datas)):
        single_img = datas[i].reshape((3, 256, 256))
        single_img = single_img.transpose((2, 1, 0))
        img_gray = cv2.cvtColor(single_img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite("gray_imgs/data_" + str(i) + ".jpg", img_gray)


if __name__ == '__main__':
    parse_imgs()

