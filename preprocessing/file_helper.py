import os
from cv2 import imread


def get_file_list(path, extension):
    """ Returns list of files in folder PATH whos extension is EXTENSION """

    extension = extension.lower()
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        break  # because only top level is wanted

    return filter(lambda f: f.lower().endswith(extension), files)


def open_image(folder, name):
    path = os.path.join(folder, name)
    img = imread(path)
    if img is None:
        print "Cant open image", path
        exit(1)
    return img
