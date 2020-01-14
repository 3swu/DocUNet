import numpy as np
from ctypes import *
from cv2 import cv2
import os

def get_c_funcs() -> dict:
    lib = np.ctypeslib.load_library('inverse_deform', '/home/wulei/DocUNet/data/c_src')
    ndpointer_float64_2d = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
    ndpointer_int32_1d   = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1)
    ndpointer_int32_2d   = np.ctypeslib.ndpointer(dtype=np.int32, ndim=2)
    
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
        ndpointer_int32_2d,
    ]

    c_get_edge = lib.get_edge
    c_get_edge.restype = None
    c_get_edge.argtypes = [
            ndpointer_float64_2d,
            ndpointer_int32_1d,
            ndpointer_int32_1d,
    ]

    c_interpolte = lib.interpolate
    c_interpolte.restype = None
    c_interpolte.argtypes = [
        ndpointer_float64_2d,
        ndpointer_float64_2d,
        ndpointer_float64_2d,
        ndpointer_int32_2d,
        ndpointer_int32_1d,
    ]

    funcs = {
        'c_inverse_deform': c_inverse_deform,
        'c_get_edge': c_get_edge,
        'c_interpolate': c_interpolte,
    }

    return funcs

def inverse_deform_padding(img_path, label: dict, funcs):
    assert os.path.exists(img_path), 'path not exists'

    label_x, label_y = label['x'], label['y']
    src_shape = label_x.shape

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_b, img_g, img_r = np.zeros(src_shape).astype(np.float64),\
        np.zeros(src_shape).astype(np.float64),\
            np.zeros(src_shape).astype(np.float64)

    img_b[:], img_g[:], img_r[:] = img[:,:,0], img[:,:,1], img[:,:,2]

    dst_shape = tuple(map(lambda x: x * 2, src_shape))

    dst_img_b, dst_img_g, dst_img_r, mask = \
        np.zeros(dst_shape).astype(np.float64),\
            np.zeros(dst_shape).astype(np.float64),\
                np.zeros(dst_shape).astype(np.float64),\
                    np.zeros(dst_shape).astype(np.int32)

    label_x, label_y = label_x.astype(np.float64), label_y.astype(np.float64)
    offset = np.array(tuple(map(lambda x : 0.5 * x, src_shape))).astype(np.int32)
    src_shape = np.array(src_shape).astype(np.int32)
    dst_shape = np.array(dst_shape).astype(np.int32)

    c_inverse_deform = funcs['c_inverse_deform']
    c_inverse_deform(label_x, label_y, src_shape, \
        img_b, img_g, img_r, dst_img_b, dst_img_g, dst_img_r, dst_shape, offset, mask)

    return dst_img_b, dst_img_g, dst_img_r, mask

def crop(dst_img_b, dst_img_g, dst_img_r, mask, funcs):
    edge = np.zeros(4,).astype(np.int32)
    dst_shape = np.array(dst_img_b.shape).astype(np.int32)

    c_get_edge = funcs['c_get_edge']
    c_get_edge(dst_img_b, dst_shape, edge)

    min_x, max_x, min_y, max_y = edge[0] + 1, edge[1] - 1, edge[2] + 1, edge[3] - 1
    dst_img_b = dst_img_b[min_x : max_x, min_y : max_y]
    dst_img_g = dst_img_g[min_x : max_x, min_y : max_y]
    dst_img_r = dst_img_r[min_x : max_x, min_y : max_y]
    mask = mask[min_x : max_x, min_y : max_y]

    return dst_img_b, dst_img_g, dst_img_r, mask

def interpolate(img_b, img_g, img_r, mask, funcs):
    c_interpolte = funcs['c_interpolate']
    img_b = np.ascontiguousarray(img_b, img_b.dtype)
    img_g = np.ascontiguousarray(img_g, img_g.dtype)
    img_r = np.ascontiguousarray(img_r, img_r.dtype)
    mask  = np.ascontiguousarray(mask , mask.dtype )
    
    shape = np.array(mask.shape).astype(np.int32)

    c_interpolte(img_b, img_g, img_r, mask, shape)

    rows, cols = mask.shape
    img_b = img_b[1 : rows - 1, 1 : cols -1]
    img_g = img_g[1 : rows - 1, 1 : cols -1]
    img_r = img_r[1 : rows - 1, 1 : cols -1]
    
    return img_b, img_g, img_r

def inverse_deform(img_path, label, save_path):
    funcs = get_c_funcs()
    dst_img_b, dst_img_g, dst_img_r, mask = inverse_deform_padding(img_path, label, funcs)
    dst_img_b, dst_img_g, dst_img_r, mask = crop(dst_img_b, dst_img_g, dst_img_r, mask, funcs)
    dst_img_b, dst_img_g, dst_img_r = interpolate(dst_img_b, dst_img_g, dst_img_r, mask, funcs)

    cv2.imwrite(save_path, np.dstack([dst_img_b, dst_img_g, dst_img_r]))


if __name__ == '__main__':
    img_path = '/home/wulei/data_generate/image/229-5574a4a0.png'
    label_path = '/home/wulei/DocUNet/model/test.npz'
    save_path = '/home/wulei/DocUNet/model/test.png'
    inverse_deform(img_path, label_path, save_path)