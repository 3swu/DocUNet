from cv2 import cv2
from ctypes import *
import numpy as np
import random
import math
import time
import os

from anti_deform import from_label_deform
from cover_texture import *

def label_visualization(label, name):
    rows, cols = label.shape

    for i in range(rows):
        for j in range(cols):
            label[i][j] = 255 if label[i][j] != 0 else 0

    cv2.imshow(name, label)


def get_random_vs(rows, cols):
    # vertex = (random.randint(int(rows / 4), int((rows / 4) * 3)), random.randint(int(cols / 4), int((cols / 4) * 3)))
    vertex = (random.randint(int(0.25 * rows), int(0.75 * rows)), random.randint(int(0.25 * cols), int(0.75 * cols)))

    avg = (rows + cols) / 2
    
    degree = 0
    while not ((degree > (1 / 12) * math.pi and degree < (5 / 12) * math.pi) or\
               (degree > (7 / 12) * math.pi and degree < (11 / 12) * math.pi) or\
               (degree > (13 / 12) * math.pi and degree < (17 / 12) * math.pi) or\
               (degree > (19 / 12) * math.pi and degree < (23 / 12) * math.pi)):
        degree = random.uniform(0, 2 * math.pi)

    v = (degree, random.uniform(avg / 12, avg / 15))
    # v = (degree, 10)
    k = math.tan(v[0])

    return vertex, v, k, avg

def distance(k, vertex, point):
    c, b, a = k * vertex[0] - vertex[1], -1 * k, 1
    return abs(a * point[0] + b * point[1] + c) / math.sqrt(a**2 + b**2)


def deform(label_shape, label_x, label_y, type):
    '''
    type:
        0 - fold
        1 - curve
    '''

    # dst_img = np.ones(src_shape) * 255

    # call C 
    c_utils = np.ctypeslib.load_library('deform_label', 'c_src')
    c_deform = c_utils.deform
    c_deform.restype = None
    c_deform.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        c_int,
    ]



    rows, cols = label_shape
    vertex, v, k, avg = get_random_vs(*label_shape)

    # prepare data
    new_label_x, new_label_y = np.zeros(label_shape).astype(np.float64), np.zeros(label_shape).astype(np.float64)
    label_x = label_x.astype(np.float64)
    label_y = label_y.astype(np.float64)
    shape = np.array([rows, cols]).astype(np.int32)
    vertex = np.array(vertex).astype(np.int32)
    v = np.array(v).astype(np.float64)


    # defrom
    c_deform(label_x, label_y, new_label_x, new_label_y, shape, vertex, v, type)

    return new_label_x, new_label_y


def gen_deform_label(img_path, resize_shape, operation : list):

    print(f'img: {os.path.abspath(img_path)}')

    # label padding
    img_shape = resize_shape
    label_shape = (int(img_shape[0] * 2), int(img_shape[1] * 2))
    label_x, label_y =  np.zeros(label_shape), np.zeros(label_shape)

    i_rows_start, i_rows_end, i_cols_start, i_cols_end = int(0.25 * label_shape[0]), \
        int(0.75 * label_shape[0]), int(0.25 * label_shape[1]), int(0.75 * label_shape[1])
    label_x[i_rows_start : i_rows_end, i_cols_start : i_cols_end] = -1
    label_y[i_rows_start : i_rows_end, i_cols_start : i_cols_end] = -1


    for i, type_ in enumerate(operation):
        print(f'round {i}, type:{"fold" if type_ == 0 else "curve"}')
        label_x, label_y = deform(label_shape, label_x, label_y, type_)        

    return label_x, label_y
