"""
Testing and checking ZCA whitening preprocessing
"""
import time
import pylab
import cv2
from preprocessing.transform_in import zca_whiten

img_path = './data/iccv09Data/images/0002136.jpg'

img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_COLOR)
# img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
if img is None:
    print "cant load image"
    exit(1)

img = img.astype('float32')
# print img
print "Img shape", img.shape, img.dtype

pylab.gray()
pylab.subplot(2, 3, 1)
pylab.axis('off')
pylab.imshow(img[:, :, 0])
pylab.subplot(2, 3, 2)
pylab.axis('off')
pylab.imshow(img[:, :, 1])
pylab.subplot(2, 3, 3)
pylab.axis('off')
pylab.imshow(img[:, :, 2])

orig_shape = img.shape
start = time.clock()
white_img = zca_whiten(img.reshape((-1, 3)))
stop = time.clock()

white_img = white_img.reshape(orig_shape)
print "Time", stop-start, "sec"
print "new image shape", white_img.shape
print "new image dtype", white_img.dtype
# print white_img[100:140, 100:140]

pylab.subplot(2, 3, 4)
pylab.axis('off')
pylab.imshow(white_img[:, :, 0])
pylab.subplot(2, 3, 5)
pylab.axis('off')
pylab.imshow(white_img[:, :, 1])
pylab.subplot(2, 3, 6)
pylab.axis('off')
pylab.imshow(white_img[:, :, 2])

pylab.show()
