from cv2 import cv2
import numpy as np
import random
import math
import time
import os

def get_random_vs(rows, cols, _):
    vertex = (random.randint(int(rows / 4), int((rows / 4) * 3)), random.randint(int(cols / 4), int((cols / 4) * 3)))
    avg = (rows + cols) / 2
    v = (random.uniform(0, 2 * math.pi), random.uniform(avg / 10, avg / 9))
    k = math.tan(v[0])

    return vertex, v, k, avg

def distance(k, vertex, point):
    c, b, a = k * vertex[0] - vertex[1], -1 * k, 1
    return abs(a * point[0] + b * point[1] + c) / math.sqrt(a**2 + b**2)


def deform(src_img, label, type):
    '''
    type:
        0 - fold
        1 - curve
    '''

    dst_img = np.ones(src_img.shape) * 255
    rows, cols, _ = src_img.shape
    vertex, v, k, avg = get_random_vs(*src_img.shape)
    distance_array_2d = np.array([distance(k, vertex, (x, y)) for x in range(rows) for y in range(cols)]).reshape((rows, cols))

    #debug
    # np.savetxt('data_gen/color_test_distance.csv', distance_array_2d, fmt='%f')

    for x in range(rows):
        for y in range(cols):
            alpha = (avg / 3) if type == 0 else 2
            w = (alpha / (distance_array_2d[x][y] + alpha)) if type == 0 else (1 - (distance_array_2d[x][y] / (rows / 2))**alpha)
            offset_x, offset_y = v[1] * math.cos(v[0]) * w, v[1] * math.sin(v[0]) * w
            src_x, src_y = x - offset_x, y - offset_y

            if (src_x < 0 or src_x >= rows - 1) or (src_y < 0 or src_y >= cols - 1):
                continue
            else:
                # label[x][y][0], label[x][y][1] = offset_x, offset_y
                label[x][y][0] += offset_x
                label[x][y][1] += offset_y

                # bilinear interpolation
                for i in range(_):
                    ceil_x, ceil_y, floor_x, floor_y = math.ceil(src_x), math.ceil(src_y), math.floor(src_x), math.floor(src_y)
                    dst_img[x][y][i] = int(src_img[floor_x][floor_y][i] * (ceil_x - src_x) * (ceil_y - src_y)\
                          + src_img[floor_x][ceil_y][i] * (ceil_x - src_x) * (src_y - floor_y)\
                          + src_img[ceil_x][floor_y][i] * (src_x - floor_x) * (ceil_y - src_y)\
                          + src_img[ceil_x][ceil_y][i] * (src_x - floor_x) * (src_y - floor_y))

    return dst_img, label

def main(img_path, data_path, operation : list):
    assert os.path.exists(img_path) and os.path.exists(data_path), 'image path or data path not exists'

    print(f'img: {os.path.abspath(img_path)}')
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    label =  np.zeros((img.shape[0], img.shape[1], 2))
    total_start = time.time()

    for i, type_ in enumerate(operation):
        start = time.time()
        print(f'round {i}, type:{"fold" if type_ == 0 else "curve"}')
        start = time.time()
        img, label = deform(img, label, type_)
        print(f'round {i} finished, time:{time.time() - start}s')

    label_path = os.path.join(data_path, 'labels')
    img_path = os.path.join(data_path, 'img')
    for path in [label_path, img_path]:
        if not os.path.exists(path):
            print(f'path {path} not exists, mkdir')
            os.mkdir(path)
    
    np.savetxt(os.path.join(label_path, filename[:filename.index('.')] + '_x.csv'), label[:,:,0], fmt='%f')
    np.savetxt(os.path.join(label_path, filename[:filename.index('.')] + '_y.csv'), label[:,:,1], fmt='%f')
    cv2.imwrite(os.path.join(img_path, filename), img)
    print(f'img: {os.path.abspath(img_path)} finished, time:{time.time() - total_start}s')



if __name__ == '__main__':
    main('data_gen/scan/6.png', 'data_gen', [0, 1, 0, 0, 1, 0])