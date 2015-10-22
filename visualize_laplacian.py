"""
Visualize laplacian pyramid with 3 layers
"""
import pylab
import cv2
from preprocessing.transform_in import get_laplacian_pyramid_layer

img_path = './data/iccv09Data/images/0002136.jpg'

img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_COLOR)
if img is None:
    print "cant load image"
    exit(1)


#   convert to YUV (inplace)
img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

print "Img shape", img.shape, img.dtype

pylab.gray()
pylab.subplot(4, 1, 1)
pylab.axis('off')
pylab.imshow(img[:, :, 0])

for i in range(3):
    pyramid_layer = get_laplacian_pyramid_layer(img, i + 1)
    pylab.subplot(4, 1, i+2)
    pylab.axis('off')
    pylab.imshow(pyramid_layer[:, :, 0])

pylab.show()
