from cv2 import cv2
import numpy as np
import os
from ctypes import *

def texture(label, src_img_b, src_img_g, src_img_r, texture_img_path):
    texture_img = cv2.imread(texture_img_path, cv2.IMREAD_COLOR)

    shape = src_img_b.shape

    texture_img_resize = cv2.resize(texture_img, (shape[1], shape[0]))
    
    texture_b, texture_g, texture_r = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    src_b, src_g, src_r = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    label_for_cover = np.zeros(shape)

    texture_b[:] = texture_img_resize[:,:,0]
    texture_g[:] = texture_img_resize[:,:,1]
    texture_r[:] = texture_img_resize[:,:,2]

    src_b[:] = src_img_b[:]
    src_g[:] = src_img_g[:]
    src_r[:] = src_img_r[:]

    label_for_cover[:] = label[:]

    src_b = src_b.astype(np.float64)
    src_g = src_g.astype(np.float64)
    src_r = src_r.astype(np.float64)
    texture_b = texture_b.astype(np.float64)
    texture_g = texture_g.astype(np.float64)
    texture_r = texture_r.astype(np.float64)
    label_for_cover = label_for_cover.astype(np.float64)

    shape = np.array(shape).astype(np.int32)

    lib = np.ctypeslib.load_library('cover_texture', 'c_src')
    c_cover_texture = lib.cover_texture
    c_cover_texture.restype = None
    c_cover_texture.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
    ]

    c_cover_texture(src_b, src_g, src_r, texture_b, texture_g, texture_r, label_for_cover, shape)

    return np.dstack([src_b, src_g, src_r])

