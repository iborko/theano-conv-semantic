# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example)
np.import_array()

# cdefine the signature of our c function
cdef extern from "superpixel.h":
    void segment_cpp(unsigned char *in_arr, int *out_arr, int width, int height,
                     float sigma, float k, int min_size)

# create the wrapper code, with numpy type annotations
def segment(np.ndarray[unsigned char, ndim=3, mode="c"] in_array not None,
            float sigma, float k, int min_size):

    if in_array.shape[2] != 3:
        raise RuntimeError("in_array must be RGB image")

    cdef int width = in_array.shape[1]
    cdef int height = in_array.shape[0]
    cdef np.ndarray[int, ndim=2] out_array = np.zeros(
        (height, width), dtype='int32')

    segment_cpp(<unsigned char*> np.PyArray_DATA(in_array),
                <int*> np.PyArray_DATA(out_array),
                width, height,
                sigma, k, min_size)

    return out_array
