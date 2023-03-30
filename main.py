import matplotlib.pyplot as plt
import numpy as np


def main():
    pass


if __name__ == '__main__':
    main()

    for fig_label in plt.get_figlabels():
        plt.figure(fig_label)
        plt.legend()
    plt.show()
