"""
Commonly used functions.
"""
import cPickle as pickle
import logging
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO

log = logging.getLogger(__name__)


def try_pickle_load(file_name, zip=None):
    """
    Tries to load pickled data from a file with
    the given name. If unsuccesful, returns None.
    Can compress using Zip.

    :param file_name: File path/name.
    :param zip: If or not the file should be zipped.
        If None, determined from file name.
    """
    if zip is None:
        zip = file_name.lower().endswith("zip")

    try:
        if zip:
            file = ZipFile(file_name, 'r')
            entry = file.namelist()[0]
            data = pickle.load(BytesIO(file.read(entry)))
        else:
            file = open(file_name, "rb")
            data = pickle.load(file)
        log.info('Succesfully loaded pickle %s', file_name)
        return data
    except IOError:
        log.info('Failed to load pickle %s', file_name)
        return None
    finally:
        if 'file' in locals():
            file.close()


def try_pickle_dump(data, file_name, zip=None, entry_name="Data.pkl"):
    """
    Pickles given data tp the given file name.
    Returns True if succesful, False otherwise.

    :param data: The object to pickle.
    :param file_name: Name of file to pickle to.
    :param zip: If or not the file should be zipped.
        If None, determined from file name.
    :param entry_name: If zipping, the name to be used
        for the ZIP entry.
    """
    if zip is None:
        zip = file_name.lower().endswith("zip")

    try:
        log.info('Attempting to pickle data to %s', file_name)
        if zip:
            file = ZipFile(file_name, 'w', ZIP_DEFLATED)
            file.writestr(entry_name, pickle.dumps(data))
        else:
            pickle.dump(data, open(file_name, "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except IOError:
        log.info('Failed to pickle data to %s', file_name)
        return False
    finally:
        if 'file' in locals():
            file.close()
