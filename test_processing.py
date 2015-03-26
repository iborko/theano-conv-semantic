import pylab
from preprocessing.class_counter import ClassCounter
from dataset.loader_msrc import load_dataset
from preprocessing.transform_in import yuv_laplacian_norm
from preprocessing.transform_out import process_out

#   load one sample
sample = None
for s in load_dataset("/media/Win/Data/MSRC_images"):
    sample = s
    break

shape = (216, 320)

# process input image
x = yuv_laplacian_norm(s.image, shape)
print "x shape", x[0].shape
print x
pylab.imshow(x[0][0])
pylab.show()

# process output image
cc = ClassCounter()
y = process_out(s.marked_image, cc, shape)
print "y shape", y.shape
print y
pylab.imshow(y)
pylab.show()
