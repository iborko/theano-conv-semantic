import os


def get_file_list(path, extension):
    """ Returns list of files in folder PATH whos extension is EXTENSION """

    extension = extension.lower()
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        break  # because only top level is wanted

    return filter(lambda f: f.lower().endswith(extension), files)
