"""
Plots cost on train and validation set.

Usage:
    python plot_cost.py output.log
"""
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) < 2:
    print 'Not enough input args. Filename required!'
    sys.exit(0)


def parse_log(fname):
    train = []
    train_acc = []
    val = []

    with open(fname) as f:
        for line in f:
            if 'training @' in line:
                train.append(float(line.split('cost')[-1].strip()))

            if 'validation cost' in line:
                val.append(float(line.split('cost:')[-1].strip()))
                train_acc.append(train)
                train = []

    train = np.vstack(map(lambda x: sum(x) / len(x), train_acc))
    val = np.vstack(val)

    return (train, val)


def main():

    train, val = parse_log(sys.argv[1])

    plt.plot(train, label='training')
    plt.plot(val, label='validation')
    plt.plot(np.ones(len(val)) * np.min(val), 'r--')
    plt.plot([np.argmin(val)], [np.min(val)], 'ro')
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    main()
