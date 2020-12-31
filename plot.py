import matplotlib.pyplot as plt
from dynamics import *

r = T0_config(10e-4, 5, 2, 0.3)


def draw_r(r):
    plt.scatter(r[:, 0], r[:, 1])
    plt.show()


draw_r(r)
