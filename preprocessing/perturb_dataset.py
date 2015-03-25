import theano
import cv2
import logging
import augment
import numpy as np
import multiprocessing as mp
import pylab

logger = logging.getLogger(__name__)


def save_result(results_img, results_marked, result):
    """
    Callback function (multiprocessing) used for dataset augmentation
    """
    i, imgs = result

    results_img[i] = imgs[0]
    results_marked[i] = imgs[1]


def perturb_img_pair(i, imgs, marked_shape):
    """
    i: int
        Ordinal number of image pair, used to save image back to 4d matrix
    imgs: image pair (2-list or 2-tuple)
        Pair of input image and marked image
    marked_out_shape: 2-tuple
        Requested downscaled shape of marked image

    This function is called in the multiprocessing pool,
    represents work that is done by one process (in multiprocessing)
    """
    image_shape = imgs[0][0].shape
    new_imgs = augment.perturb_image(imgs, image_shape)
    marked_downscaled = cv2.resize(
        new_imgs[1], (marked_shape[1], marked_shape[0]),
        interpolation=cv2.INTER_NEAREST)
    new_imgs[1] = marked_downscaled
    return i, new_imgs


def change_train_set(shared_x, x, shared_y, y, marked_shape):
    """
    Generates perturbed dataset. Dataset is pair of input images,
    and marked images. Output is put to shared variables.
    Marked images are resized to marked_shape.

    shared_x: theano shared variable
        Shared variable for storing input images
    x: numpy array (nimages, channels, y, x)
        Array of input images
    shared_y: theano shared variable
        Shared variable for marked images
    y: numpy array (nimages, y, x)
        Array of marked images
    marked_shape: 2-tuple
        Shape to which marked images have to resized
    """
    new_x = np.zeros_like(shared_x.get_value(),
                          dtype=theano.config.floatX)
    new_y_shape = (y.shape[0], marked_shape[0], marked_shape[1])
    new_y = np.zeros(new_y_shape,
                     dtype=theano.config.floatX)
    logger.info("\|/- Started perturbing dataset")

    pool = mp.Pool(mp.cpu_count())

    result_func = lambda result: save_result(new_x, new_y, result)

    for i in xrange(len(x)):
        in_list = [x[i], y[i]]
        pool.apply_async(perturb_img_pair, args=(i, in_list, marked_shape,),
                         callback=result_func)
        # result_func(perturb_img_pair(i, in_list, marked_shape))  # no MP
    pool.close()
    pool.join()

    shared_x.set_value(new_x, borrow=True)
    shared_y.set_value(new_y.reshape((new_y.shape[0], -1)), borrow=True)

    '''
    # debugging purposes
    for j in xrange(5):
        pylab.subplot(2, 5, 0 * 5 + j + 1)
        pylab.axis('off')
        pylab.imshow(new_x[j, 0, :, :])
        pylab.gray()

        pylab.subplot(2, 5, 1 * 5 + j + 1)
        pylab.axis('off')
        pylab.imshow(new_y[j, :, :])
        pylab.jet()
    pylab.show()
    '''

    logger.info("\|/- Done perturbing dataset")
