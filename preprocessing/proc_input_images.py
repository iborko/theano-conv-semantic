import sys
import cv2
import numpy as np
import cPickle as pickle

from file_helper import get_file_list

EXTENSION = '.bmp'


def create_empty_img(dim):
    height, width = dim
    blank_image = np.zeros((height, width, 3), np.uint8)
    return blank_image


def get_laplacian_pyramid_layer(img, n):
    '''Returns the n-th layer of the laplacian pyramid'''
    currImg, i = img, 0
    while i < n:
        down = cv2.pyrDown(currImg)
        up = cv2.pyrUp(down)
        lap = currImg - up
        currImg = down
        i += 1
    return lap


def normalize_block(block):
    mean = np.mean(block, dtype=np.float64)
    block -= mean

    stddev = np.std(block, dtype=np.float64)
    if stddev > 0.000001:
        block /= stddev


def normalize(img):
    ''' Normalize img in blocks of 15x15 '''
    b_x, b_y = 15, 15
    for x in xrange(0, img.shape[0], b_x):
        for y in xrange(0, img.shape[1], b_y):
            normalize_block(img[x:(x + b_x), y:(y + b_y)])


def process_images(inpath, outpath):

    img_list = get_file_list(inpath, EXTENSION)
    img_list.sort()
    with open("infiles.log", 'w') as f:
        f.writelines("\n".join(img_list))

    def_shape = (216, 320)
    images = np.zeros((len(img_list), def_shape[0], def_shape[1], 3),
                      dtype=np.uint8)
    print "Images shape", images.shape

    # load images and fill with rgb(0,0,0) until def_shape
    for idx, img_name in enumerate(img_list):
        img = cv2.imread(inpath + img_name, cv2.CV_LOAD_IMAGE_COLOR)
        if img is None:
            print "Cant open image", img_name
            exit(1)

        print img_name, "has shape", img.shape
        # rotate image for 90, if in portrait orientation
        if img.shape[0] > img.shape[1]:
            img = cv2.transpose(img)

        # crop to def_shape shape
        img = img[:def_shape[0], :def_shape[1], :]

        fill_y = def_shape[0] - img.shape[0]
        fill_x = def_shape[1] - img.shape[1]
        if fill_x < 0 or fill_y < 0:
            print "Default image shape not valid", img.shape
            exit(1)
        img = cv2.copyMakeBorder(img, fill_x, fill_y, 0, 0,
                                 cv2.BORDER_CONSTANT, 0)
        images[idx] = img

    print "Starting YUV conversion"
    # convert to YUV (inplace)
    for i in xrange(len(img_list)):
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2YUV)

    print "Building lapacian pyramids"
    # generate laplacian pyramid (level 1)
    pyr_level1 = np.zeros(images.shape, dtype=np.float32)
    for i in xrange(len(img_list)):
        pyr_level1[i] = get_laplacian_pyramid_layer(images[i], 1) / 255.0

    print "Normalizing blocks"
    # normalize every channel
    for i in xrange(len(img_list)):
        normalize(pyr_level1[i, :, :, 0])  # Y
        normalize(pyr_level1[i, :, :, 1])  # U
        normalize(pyr_level1[i, :, :, 2])  # V

    cv2.imshow('laplace1', pyr_level1[0, :, :, 0])
    cv2.waitKey(0)

    # print "Before axis swap\n", pyr_level1[0, :5, :5, 0]
    pyr_level1 = np.rollaxis(pyr_level1, 3, 1)
    # print "After axis swap\n", pyr_level1[0, 0, :5, :5]

    print "Writing matrices to output file"
    with open(outpath, 'wb') as f:
        pickle.dump(pyr_level1, f, protocol=pickle.HIGHEST_PROTOCOL)

    del pyr_level1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Wrong number of arguments, 3 needed"
        print __file__, "<input_path> <output_file>"
        exit(1)

    process_images(sys.argv[1], sys.argv[2])
