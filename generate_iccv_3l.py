"""
Generates input and output numpy data for both train and test set, given
path to dataset.
Data is pickled in OUT_PATH folder, in .bin files.
"""
import sys
import logging

from preprocessing.transform_in import yuv_laplacian_norm
from helpers.load_conf import load_config
from generate_iccv_1l import main

logger = logging.getLogger(__name__)

requested_shape = (240, 320)


def gen_layers_for_image(i, img):
    #   3 layers, lapacian pyramid
    layers = yuv_laplacian_norm(img, requested_shape, 3)
    return i, layers


if __name__ == "__main__":
    '''
    python generate_iccv_3l.py gen.conf [show]
    '''
    logging.basicConfig(level=logging.INFO)

    show = False
    argc = len(sys.argv)
    if argc == 2:
        conf_path = sys.argv[1]
    elif argc == 3:
        conf_path = sys.argv[1]
        if sys.argv[2] == "show":
            show = True
        else:
            print "Wrong arguments"
            exit(1)
    else:
        print "Too few arguments"
        exit(1)

    conf = load_config(conf_path)
    if conf is None:
        exit(1)

    main(conf, gen_layers_for_image, n_layers=3, show=show)
