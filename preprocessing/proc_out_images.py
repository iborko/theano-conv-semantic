from threading import Lock
import sys
import cPickle as pickle
import numpy as np
import cv2

from file_helper import get_file_list

EXTENSION = '.bmp'
OUT_SHAPE = (43, 69)


class ClassCounter(object):
    """ class used to transform color matrix to index matrix """

    def __init__(self):
        self.classes = {}
        self.index = 0
        self.lock = Lock()
        self.get_num((0, 0, 0))

    def count_matrix(self, m):
        """ run class counting on 2d matrix """
        for i in xrange(len(m)):
            for j in xrange(len(m[i])):
                if type(m[i, j]) is not tuple:
                    m[i, j] = self.get_num(tuple(m[i, j]))
                else:
                    m[i, j] = self.get_num(m[i, j])
        return m

    def get_num(self, color):
        """ get the number representing specific color """
        with self.lock:
            if color not in self.classes:
                self.classes[color] = self.index
                self.index += 1
        return self.classes[color]

    def get_total_colors(self):
        """ return the number of different colors """
        return self.index


def process_images(inpath, outpath):

    img_list = get_file_list(inpath, EXTENSION)
    # img_list = img_list[:10]  # DEBUG
    img_list.sort()
    with open("outfiles.log", 'w') as f:
        f.writelines("\n".join(img_list))

    def_shape = (216, 320)  # this is image shape
    images = np.zeros((len(img_list), OUT_SHAPE[0], OUT_SHAPE[1]),
                      dtype=np.uint8)
    print "Images shape", images.shape

    cc = ClassCounter()

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

        # crop to def_shape
        img = img[:def_shape[0], :def_shape[1], :]

        fill_y = def_shape[0] - img.shape[0]
        fill_x = def_shape[1] - img.shape[1]
        if fill_x < 0 or fill_y < 0:
            print "Default image shape not valid", img.shape
            exit(1)
        img = cv2.copyMakeBorder(img, fill_x, fill_y, 0, 0,
                                 cv2.BORDER_CONSTANT, 0)
        img = cc.count_matrix(img)  # gives (r, g, b) for every pixel, where r=g=b=class_index
        img = img[:, :, 0]
        img = cv2.resize(img, (OUT_SHAPE[1], OUT_SHAPE[0]),
                         interpolation=cv2.INTER_NEAREST)
        images[idx] = img

    print "Found %d classes" % (cc.get_total_colors())
    print "Classes", cc.classes

    print "Writing matrices to file"
    with open(outpath, 'wb') as f:
        pickle.dump(images, f, protocol=pickle.HIGHEST_PROTOCOL)


def test_class_counter():
    cc = ClassCounter()

    data = np.random.randint(5, 8, (2, 3, 2))
    print "data\n", data

    result = cc.count_matrix(data)
    print "result\n", result[:, :, 0]

    print "Found %d different colors" % (cc.get_total_colors())


if __name__ == "__main__":
    """
    Usage:
        python proc_out_images.py <in_path> <out_file>
    Example:
        python proc_out_images.py data/cows/seg/ data_y.bin
    """
    argc = len(sys.argv)
    if argc == 3:
        in_path = sys.argv[1]
        out_path = sys.argv[2]
        process_images(in_path, out_path)
    else:
        print "Wrong arguments"
    # test_class_counter()
