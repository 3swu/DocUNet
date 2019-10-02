import deform
from ctypes import *
import time
import numpy as np

if __name__ == '__main__':
    # vertex, _, k, _ = deform.get_random_vs(1000, 1000, None)
    # vertex = (261, 392)
    # k = 1.0842066
    # print(f'vertex: {vertex} , k: {k}')
    # print(f'python result: {deform.distance(k, vertex, (500, 500))}')


    # # c
    # c_utils = CDLL('code/c_src/util.so')
    # c_k = c_float(k)
    # c_distance = c_utils.distance
    # c_distance.restype = c_float
    # print(f'c result: {c_distance(c_k, vertex[0], vertex[1], 500, 500)}')

    # distance_xy = 800.34
    # alpha = 300
    # rows = 2030
    # type = 0

    # result = (alpha / (distance_xy + alpha)) if type == 0 else (1 - (distance_xy / (rows / 2))**alpha)
    # print(f'debug: {(distance_xy / (rows / 2)) ** alpha}')
    # print(f'python result: {result}')

    # c_utils = CDLL('code/c_src/util.so')
    # c_w = c_utils.w
    # c_w.restype = c_float

    # print(f'c result: {c_w(c_float(alpha), c_float(distance_xy), rows, type)}')

    # a = np.load('/home/wulei/DocUNet/data_gen/labels/dtd_0062.npy')
    # a = np.random.rand(4, 4, 2)
    a = np.array(range(16)).reshape((4, 4))
    b = np.array(range(16, 32)).reshape((4, 4))
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    # a = np.array(range(16)).reshape((4, 4))
    # a.dtype = np.float64
    print(a)
    print(b)
    print('----')
    lib = np.ctypeslib.load_library('util', 'c_src')
    c_deform = lib.deform
    c_deform.restype = None
    

    rows, cols = a.shape
    shape = np.array(a.shape)
    vertex = np.array([2, 3])
    v = np.array((2.8234, 1.4343)).astype(np.float32)

    c_deform.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2),
        np.ctypeslib.ndpointer(dtype=np.int, ndim=1),
        np.ctypeslib.ndpointer(dtype=np.int, ndim=1),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1),
        c_int,
    ]

    

    # pointer = a.ctypes.data_as(c_void_p)
    # dataptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    c_deform(a, b, shape, vertex, v, 1)

    # print(a[:,0,:])
    # print(a[:,:,1])

    print(a)
    print(b)
