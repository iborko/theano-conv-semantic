import os


def get_file_list(path, extension):
    """ Returns list of files in folder PATH whos extension is EXTENSION """

    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        break  # because only top level is wanted

    files = [f for f in files if f.endswith(extension)]
    return files
