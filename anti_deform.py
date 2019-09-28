from cv2 import cv2
import numpy as np
import time
import os
import math

def from_label_deform(label, img_path):
    assert os.path.exists(img_path)

    # label_x = np.loadtxt(label_csv_path[0])
    # label_y = np.loadtxt(label_csv_path[1])
    # label = np.load(label_path)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    rows, cols, _ = img.shape
    assert (rows, cols, 2) == label.shape, 'shape of labels and img is not same'

    dst_img = np.ones(img.shape) * 255

    for x in range(rows):
        for y in range(cols):
            src_x = x - label[x][y][0]
            src_y = y - label[x][y][1]

            if(src_x < 0 or src_x >= rows - 1) or (src_y < 0 or src_y >= cols - 1):
                continue
            else:

                # debug
                # print(f'x: {x}  y : {y} label_x: {label_x[x][y]} label_y: {label_y[x][y]}')
                # print(f'dst_x : {dst_x}  dst_y: {dst_y}')
                # print('---')

                for i in range(_):
                    ceil_x, ceil_y, floor_x, floor_y = math.ceil(src_x), math.ceil(src_y), math.floor(src_x), math.floor(src_y)
                    dst_img[x][y][i] = int(img[floor_x][floor_y][i] * (ceil_x - src_x) * (ceil_y - src_y)\
                          + img[floor_x][ceil_y][i] * (ceil_x - src_x) * (src_y - floor_y)\
                          + img[ceil_x][floor_y][i] * (src_x - floor_x) * (ceil_y - src_y)\
                          + img[ceil_x][ceil_y][i] * (src_x - floor_x) * (src_y - floor_y))
    
    return dst_img

if __name__ == '__main__':
    start = time.time()
    img = from_label_deform('data_gen/labels/dtd_0062.npy', 'data_gen/scan/dtd_0062.jpg')
    cv2.imwrite('data_gen/img/dtd_0062_anti.jpg', img)
    print(f'time:{time.time() - start}')