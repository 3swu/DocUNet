from cv2 import cv2
import numpy as np
import time
import os
import math

def from_label_deform(label_x, label_y, img_path):
    assert os.path.exists(img_path)

    # label_x = np.loadtxt(label_csv_path[0])
    # label_y = np.loadtxt(label_csv_path[1])
    # label = np.load(label_path)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    lib = np.ctypeslib.load_library('from_label_deform', 'c_src')

    rows, cols, _ = img.shape
    # assert (rows, cols) == label_x.shape and (rows, cols) == label_y.shape, 'shape of labels and img is not same'

    src_shape = np.array([rows, cols]).astype(np.int32)
    dst_shape = np.array(label_x.shape).astype(np.int32)

    image_edge = np.zeros(4, ).astype(np.int32)


    
    dst_img_b = np.zeros(label_x.shape)
    dst_img_g = np.zeros(label_x.shape)
    dst_img_r = np.zeros(label_x.shape)

    src_img_b = np.zeros((rows, cols))
    src_img_g = np.zeros((rows, cols))
    src_img_r = np.zeros((rows, cols))

    src_img_b[:] = img[:,:,0]
    src_img_g[:] = img[:,:,1]
    src_img_r[:] = img[:,:,2]

    
    c_from_label_deform = lib.from_label_deform
    c_from_label_deform.restype = None
    c_from_label_deform.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1),
    ]

    
    # img = img.astype(np.float32)
    dst_img_b = dst_img_b.astype(np.float64)
    dst_img_g = dst_img_g.astype(np.float64)
    dst_img_r = dst_img_r.astype(np.float64)
    src_img_b = src_img_b.astype(np.float64)
    src_img_g = src_img_g.astype(np.float64)
    src_img_r = src_img_r.astype(np.float64)

    c_from_label_deform(label_x, label_y, dst_shape, src_img_b, src_img_g, src_img_r, dst_img_b, dst_img_g, dst_img_r, src_shape, image_edge)

    min_row, max_row, min_col, max_col = np.asarray(image_edge)
    min_row = min_row - 50 if min_row - 50 > 0 else 0
    max_row = max_row + 50 if max_row + 50 < dst_shape[0] else dst_shape[0]
    min_col = min_col - 50 if min_col - 50 > 0 else 0
    max_col = max_col + 50 if max_col + 50 < dst_shape[1] else dst_shape[1]

    dst_img_b = dst_img_b[min_row : max_row, min_col : max_col]
    dst_img_g = dst_img_g[min_row : max_row, min_col : max_col]
    dst_img_r = dst_img_r[min_row : max_row, min_col : max_col]

    return np.dstack([dst_img_b, dst_img_g, dst_img_r])



if __name__ == '__main__':
    start = time.time()
    img = from_label_deform('data_gen/labels/dtd_0062.npy', 'test(npy)', 'data_gen/scan/dtd_0062.jpg')
    cv2.imwrite('data_gen/img/dtd_0062_anti.jpg', img)
    print(f'time:{time.time() - start}')