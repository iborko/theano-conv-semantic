import time
import pylab
import cv2
from superpixel import segment

img_path = '../data/iccv09Data/images/0002136.jpg'

img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_COLOR)
print "Img shape", img.shape, img.dtype

pylab.subplot(2, 1, 1)
pylab.axis('off')
pylab.imshow(img)

start = time.clock()
marks = segment(img, 0.5, 250, 200)
stop = time.clock()
print "Time", stop-start, "sec"
print "Marked image shape", marks.shape
# print marks[100:140, 100:140]

'''
colors = {}
index = 0
for i in xrange(marks.shape[0]):
    for j in xrange(marks.shape[1]):
        current_mark = marks[i, j]
        if current_mark not in colors:
            colors[current_mark] = index
            index += 1
        marks[i, j] = colors[current_mark]
print marks[100:140, 100:140]
'''

pylab.subplot(2, 1, 2)
pylab.axis('off')
pylab.imshow(marks)

pylab.show()
