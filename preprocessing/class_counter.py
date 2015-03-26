import numpy as np
import logging
import multiprocessing as mp

log = logging.getLogger(__name__)


class ClassCounter(object):
    '''
    class used to transform color matrix to index matrix
    '''

    def __init__(self):
        manager = mp.Manager()
        self.classes = manager.dict()
        self.lock = manager.Lock()
        self.index = manager.Value('i', 0)
        self.get_num((0, 0, 0))
        log.info("Class counter initialized")

    def count_matrix(self, m):
        ''' run class counting on 2d matrix '''
        # find all colors
        set_colors = set([])
        for i in xrange(len(m)):
            for j in xrange(len(m[i])):
                    set_colors.add(tuple(m[i, j]))
        # find numbers for colors
        for color in set_colors:
            self.get_num(color)
        
        # now we have all colors, just set them
        out_matrix = np.zeros((m.shape[0], m.shape[1]), dtype='int32')
        for i in xrange(len(m)):
            for j in xrange(len(m[i])):
                out_matrix[i, j] = self.classes[tuple(m[i, j])]
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

    def log_stats(self):
        """
        Logs with logger (logging level INFO) data about Class counter.
        Logger must me initialized.
        """
        log.info("Found %d classes", self.get_total_colors())
        log.info("Classes %s", self.classes)


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
