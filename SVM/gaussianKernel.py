import numpy as np


def gaussian_kernel(x1, x2, sigma):
    x1 = x1.flatten() #z转换为1维数组
    x2 = x2.flatten()

    sim = 0

    sim=np.exp(((x1-x2).dot(x1-x2))/(-2*sigma*sigma))

    return sim

