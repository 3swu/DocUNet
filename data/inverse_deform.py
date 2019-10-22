import numpy as np
from ctypes import *
from cv2 import cv2
import os

def get_c_funcs() -> dict:
    lib = np.ctypeslib.load_library('inverse_deform', '.')
    ndpointer_float64_2d = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
    ndpointer_int32_1d   = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1)
    
    c_inverse_deform = lib.inverse_deform
    c_inverse_deform.restype = None
    c_inverse_deform.argtypes = [
        ndpointer_float64_2d,
        ndpointer_float64_2d,
        ndpointer_int32_1d,
        ndpointer_float64_2d,
        ndpointer_float64_2d,
        ndpointer_float64_2d,
        ndpointer_float64_2d,
        ndpointer_float64_2d,
        ndpointer_float64_2d,
        ndpointer_int32_1d,
        ndpointer_int32_1d,
    ]

    c_get_edge = lib.get_edge
    c_get_edge.restype = None
    c_get_edge.argtypes = [
            ndpointer_float64_2d,
            ndpointer_int32_1d,
            ndpointer_int32_1d,
    ]

    funcs = {
        'c_inverse_deform': c_inverse_deform,
        'c_get_edge': c_get_edge,
    }

    return funcs

def inverse_deform_padding(img_path, label_path, funcs):
    assert os.path.exists(img_path) and os.path.exists(label_path), 'path not exists'

    label = np.load(label_path)
    label_x, label_y = label['x'], label['y']
    src_shape = label_x.shape

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_b, img_g, img_r = np.zeros(src_shape).astype(np.float64),\
        np.zeros(src_shape).astype(np.float64),\
            np.zeros(src_shape).astype(np.float64)

    img_b[:], img_g[:], img_r[:] = img[:,:,0], img[:,:,1], img[:,:,2]

    dst_shape = tuple(map(lambda x: x * 2, src_shape))

    dst_img_b, dst_img_g, dst_img_r = \
        np.zeros(dst_shape).astype(np.float64),\
            np.zeros(dst_shape).astype(np.float64),\
                np.zeros(dst_shape).astype(np.float64)

    label_x, label_y = label_x.astype(np.float64), label_y.astype(np.float64)
    offset = np.array(tuple(map(lambda x : 0.5 * x, src_shape))).astype(np.int32)
    src_shape = np.array(src_shape).astype(np.int32)
    dst_shape = np.array(dst_shape).astype(np.int32)

    c_inverse_deform = funcs['c_inverse_deform']
    c_inverse_deform(label_x, label_y, src_shape, \
        img_b, img_g, img_r, dst_img_b, dst_img_g, dst_img_r, dst_shape, offset)

    return dst_img_b, dst_img_g, dst_img_r

def crop(dst_img_b, dst_img_g, dst_img_r, funcs):
    edge = np.zeros(4,).astype(np.int32)
    dst_shape = np.array(dst_img_b.shape).astype(np.int32)

    c_get_edge = funcs['c_get_edge']
    c_get_edge(dst_img_b, dst_shape, edge)

    min_x, max_x, min_y, max_y = edge[0] + 2, edge[1] - 2, edge[2] + 2, edge[3] - 2
    dst_img_b = dst_img_b[min_x : max_x, min_y : max_y]
    dst_img_g = dst_img_g[min_x : max_x, min_y : max_y]
    dst_img_r = dst_img_r[min_x : max_x, min_y : max_y]

    return dst_img_b, dst_img_g, dst_img_r





