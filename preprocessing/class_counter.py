import numpy as np
import logging
import multiprocessing as mp

log = logging.getLogger(__name__)

# maximum color value
MAX_VAL = 256


class ClassCounter(object):
    '''
    class used to transform color matrix to index matrix
    '''

    def __init__(self):
        manager = mp.Manager()
        self.classes = manager.dict()
        self.lock = manager.Lock()
        self.index = manager.Value('i', 0)
        self.get_num(0)
        log.info("Class counter initialized")

    def count_matrix(self, m):
        ''' run class counting on 2d matrix '''

        # convert (r,g,b) tuples to discrete numbers
        converted_m = np.dot(m, np.array([MAX_VAL**2, MAX_VAL, 1]))
        # find all colors
        unique_colors = np.unique(converted_m)

        # find numbers for colors
        for color in unique_colors:
            self.get_num(color)

        # copy manager.dict() (which is mp-safe) to local dict
        classes_copy = {}
        classes_copy.update(self.classes)

        # now we have all colors, just set them
        out_matrix = np.zeros((m.shape[0], m.shape[1]), dtype='int32')
        for k, v in classes_copy.iteritems():
            out_matrix[converted_m==k] = v
        return out_matrix

    def get_num(self, color):
        ''' get the number representing specific color '''
        with self.lock:
            if color not in self.classes:
                self.classes[color] = self.index.value
                self.index.value += 1
        return self.classes[color]

    def get_total_colors(self):
        """ return the number of different colors """
        return self.index.value

    def _convert_to_rgb(self, value):
        """
        value: int
            discrete number representing RGB tuple

        Returns: 3-tuple
            R, G, B componenets
        """
        r = value // MAX_VAL**2
        g = (value - r * MAX_VAL**2) // MAX_VAL
        b = value - r * MAX_VAL**2 - g * MAX_VAL
        return (r, g, b)

    def log_stats(self):
        """
        Logs with logger (logging level INFO) data about Class counter.
        Logger must me initialized.
        """
        rgb_dict = {}
        for key in self.classes.keys():
            rgb_dict[self.classes[key]] = self._convert_to_rgb(key)
        log.info("Found %d classes", self.get_total_colors())
        log.info("Classes %s", rgb_dict)


def test_class_counter():
    logging.basicConfig(level=logging.DEBUG)
    cc = ClassCounter()

    data = np.random.randint(5, 8, (2, 3, 2))
    print "data\n", data

    result = cc.count_matrix(data)
    print "result\n", result

    cc.log_stats()

if __name__ == "__main__":
    test_class_counter()
