'''
Module for testing of dataset perturbations
'''

import pylab
import sys
import numpy as np
from preprocessing.augment import perturb_image
from util import try_pickle_load

IMAGES_TO_SHOW = 5


def main(path, marked_path=None):
    imgs_allscales = try_pickle_load(path)
    n_scales = len(imgs_allscales)
    imgs = imgs_allscales[0]  # scale 1
    image_shape = (imgs.shape[2], imgs.shape[3])
    
    images_to_show = min(IMAGES_TO_SHOW, len(imgs))

    print "Images shape", imgs.shape
    print "Number of images to show", images_to_show
    print "Number of scales", n_scales
    print "Requested image shape will be", image_shape

    perturbed_imgs = np.empty((images_to_show, imgs.shape[1],
                               imgs.shape[2], imgs.shape[3]))
    perturbed_marks = None
    if marked_path is not None:
        marked_imgs = try_pickle_load(marked_path)
        perturbed_marks = np.empty((images_to_show, marked_imgs.shape[1],
                                    marked_imgs.shape[2]))

    for i in xrange(images_to_show):
        if perturbed_marks is None:
            ret_img = perturb_image(imgs[i], image_shape)
            perturbed_imgs[i] = ret_img
        else:
            ret_list = perturb_image([imgs[i], marked_imgs[i]], image_shape)
            perturbed_imgs[i] = ret_list[0]
            perturbed_marks[i] = ret_list[1]

    for j in xrange(images_to_show):
        pylab.subplot(4, images_to_show, 0 * images_to_show + j + 1)
        pylab.axis('off')
        pylab.imshow(imgs[j, 0, :, :])
        pylab.gray()  # set colormap
        
        pylab.subplot(4, images_to_show, 1 * images_to_show + j + 1)
        pylab.axis('off')
        pylab.imshow(perturbed_imgs[j, 0, :, :])
        pylab.gray()

    if perturbed_marks is not None:
        for j in xrange(images_to_show):
            pylab.subplot(4, images_to_show, 2 * images_to_show + j + 1)
            pylab.axis('off')
            pylab.imshow(marked_imgs[j, :, :])
            pylab.jet()
            
            pylab.subplot(4, images_to_show, 3 * images_to_show + j + 1)
            pylab.axis('off')
            pylab.imshow(perturbed_marks[j, :, :])
            pylab.jet()

    pylab.show()

if __name__ == "__main__":
    """
    Example of run command:
    python test_data_perturbations.py data/x_train.bin
    """
    argc = len(sys.argv)
    if argc == 2:
        main(sys.argv[1])
    elif argc == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print "Wrong arguments"
        exit(1)
