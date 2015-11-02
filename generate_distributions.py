"""
Given marked images (output classes), distributions of classes for pixel
neighborhood are calculated.
"""

import numpy as np

sample = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])


def img_distr(img, size):
    """
    Calc class distribution for pixel neighborhood
    """
    assert(img.ndim == 2)
    img_size_x = img.shape[0]
    img_size_y = img.shape[1]

    n_classes = img.max() + 1

    count = np.zeros((n_classes, sample.shape[0], sample.shape[1]), np.int32)
    total = np.zeros(sample.shape, np.int32)

    left_offset = size // 2
    right_offset = size - left_offset - 1

    for i in xrange(img_size_x):
        for j in xrange(img_size_y):
            for k in xrange(-left_offset, right_offset + 1):
                for l in xrange(-left_offset, right_offset + 1):
                    cx = i + k
                    cy = j + l
                    if cx < 0 or cx >= img_size_x:
                        continue
                    if cy < 0 or cy >= img_size_y:
                        continue

                    c = img[cx, cy]
                    total[i, j] += 1
                    count[c, i, j] += 1

    distr = count.astype('float32') / total

    return distr


print "sample", sample
print "distr", img_distr(sample, 3)
