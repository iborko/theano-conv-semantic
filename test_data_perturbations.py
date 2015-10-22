'''
Module for testing of dataset perturbations
'''

import pylab
import sys
import numpy as np
from preprocessing.augment import perturb_image
from util import try_pickle_load

IMAGES_TO_SHOW = 5
CHANNEL = 0  # channel to show


def main(path, marked_path=None):
    # images multiscale
    imgs_mscale = try_pickle_load(path)
    n_scales = len(imgs_mscale)
    imgs_s0 = imgs_mscale[0]  # scale 1
    image_shape = (imgs_s0.shape[2], imgs_s0.shape[3])

    images_to_show = min(IMAGES_TO_SHOW, len(imgs_s0))

    print "Images shape", imgs_s0.shape
    print "Number of images to show", images_to_show
    print "Number of scales", n_scales
    print "Requested image shape will be", image_shape
    n_rows = (1 + n_scales) * 2

    perturbed_imgs = [np.empty((images_to_show, imgs.shape[1],
                                imgs.shape[2], imgs.shape[3]))
                      for imgs in imgs_mscale]
    perturbed_marks = None
    if marked_path is not None:
        marked_imgs = try_pickle_load(marked_path)
        perturbed_marks = np.empty((images_to_show, marked_imgs.shape[1],
                                    marked_imgs.shape[2]))

    for i in xrange(images_to_show):
        imgs_to_perturb = [img[i] for img in imgs_mscale]
        # if we loaded markings, add marked image to list of imgs to perturb
        if perturbed_marks is not None:
            imgs_to_perturb.append(marked_imgs[i])

        ret_list = perturb_image(imgs_to_perturb, image_shape)
        for n_scale in range(n_scales):
            perturbed_imgs[n_scale][i] = ret_list[n_scale]

        if perturbed_marks is not None:
            perturbed_marks[i] = ret_list[n_scales]

    for i, imgs in enumerate(imgs_mscale):
        for j in xrange(images_to_show):
            pylab.subplot(n_rows, images_to_show, i * images_to_show + j + 1)
            pylab.axis('off')
            pylab.imshow(imgs[j, CHANNEL, :, :])
            pylab.gray()  # set colormap

    for ind, imgs in enumerate(perturbed_imgs):
        i = n_scales + ind
        for j in xrange(images_to_show):
            pylab.subplot(n_rows, images_to_show, i * images_to_show + j + 1)
            pylab.axis('off')
            pylab.imshow(imgs[j, CHANNEL, :, :])
            pylab.gray()

    if perturbed_marks is not None:
        for j in xrange(images_to_show):
            pylab.subplot(n_rows, images_to_show, (2*n_scales+0) * images_to_show + j + 1)
            pylab.axis('off')
            pylab.imshow(marked_imgs[j, :, :])
            pylab.jet()

            pylab.subplot(n_rows, images_to_show, (2*n_scales+1) * images_to_show + j + 1)
            pylab.axis('off')
            pylab.imshow(perturbed_marks[j, :, :])
            pylab.jet()

    pylab.show()

if __name__ == "__main__":
    """
    Example of run command:
    python test_data_perturbations.py data/x_train.bin
    python test_data_perturbations.py data/x_train.bin data/y_train.bin
    """
    argc = len(sys.argv)
    if argc == 2:
        main(sys.argv[1])
    elif argc == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print "Wrong arguments"
        exit(1)
