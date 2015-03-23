import numpy as np
from threading import Lock


class ClassCounter(object):
    '''
    class used to transform color matrix to index matrix
    '''

    def __init__(self):
        self.classes = {}
        self.index = 0
        self.lock = Lock()
        self.get_num((0, 0, 0))

    def count_matrix(self, m):
        ''' run class counting on 2d matrix '''
        for i in xrange(len(m)):
            for j in xrange(len(m[i])):
                if type(m[i, j]) is not tuple:
                    m[i, j] = self.get_num(tuple(m[i, j]))
                else:
                    m[i, j] = self.get_num(m[i, j])
        return m

    def get_num(self, color):
        ''' get the number representing specific color '''
        with self.lock:
            if color not in self.classes:
                self.classes[color] = self.index
                self.index += 1
        return self.classes[color]

    def get_total_colors(self):
        """ return the number of different colors """
        return self.index


def test_class_counter():
    cc = ClassCounter()

    data = np.random.randint(5, 8, (2, 3, 2))
    print "data\n", data

    result = cc.count_matrix(data)
    print "result\n", result[:, :, 0]

    print "Found %d different colors" % (cc.get_total_colors())


if __name__ == "__main__":
    test_class_counter()
