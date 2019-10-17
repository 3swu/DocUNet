from ctypes import *
import numpy as np

def resize_img_label(label_x, label_y, img_b, img_g, img_r, resize_shape):
    lib = np.ctypeslib.load_library('resize_img_label', 'c_src')
    c_resize_img_label = lib.resize_img_label
    c_resize_img_label.restype = None
    ndpointer_float64_2d = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
    ndpointer_int32_1d = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1)
    c_resize_img_label.argtypes = [
        ndpointer_float64_2d,
        ndpointer_float64_2d,
        ndpointer_float64_2d,
        ndpointer_float64_2d,
        ndpointer_float64_2d,
        ndpointer_float64_2d,
        ndpointer_float64_2d,
        ndpointer_float64_2d,
        ndpointer_float64_2d,
        ndpointer_float64_2d,
        ndpointer_int32_1d,
        ndpointer_int32_1d,  
    ]

    new_img_b, new_img_g, new_img_r, new_label_x, new_label_y = \
        np.zeros(resize_shape).astype(np.float64), np.zeros(resize_shape).astype(np.float64), \
            np.zeros(resize_shape).astype(np.float64), np.zeros(resize_shape).astype(np.float64), \
                np.zeros(resize_shape).astype(np.float64)

    # copy label_x and label_y for the flag 'C_CONTIGUOUS'
    temp_label_x, temp_label_y = label_x, label_y
    label_x, label_y = np.zeros(temp_label_x.shape), np.zeros(temp_label_x.shape)
    label_x[:], label_y[:] = temp_label_x[:], temp_label_y[:]
    label_x, label_y = label_x.astype(np.float64), label_y.astype(np.float64)
    
    original_shape = np.array(label_x.shape).astype(np.int32)
    new_shape = np.array(resize_shape).astype(np.int32)

    # make sure that the image is c contiguous
    if not img_b.flags['C_CONTIGUOUS']:
        img_b = np.ascontiguousarray(img_b, dtype=img_b.dtype)
    if not img_g.flags['C_CONTIGUOUS']:
        img_g = np.ascontiguousarray(img_g, dtype=img_g.dtype)
    if not img_r.flags['C_CONTIGUOUS']:
        img_r = np.ascontiguousarray(img_r, dtype=img_r.dtype)
    
    c_resize_img_label(label_x, label_y, new_label_x, new_label_y,\
        img_b, img_g, img_r, new_img_b, new_img_g, new_img_r, original_shape, new_shape)

    return new_label_x, new_label_y, new_img_b, new_img_g, new_img_r