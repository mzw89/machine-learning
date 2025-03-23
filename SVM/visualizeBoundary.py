import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets


def visualize_boundary(clf, X, x_min, x_max, y_min, y_max): #x,y轴的取值范围
    h = .02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))#在x，y轴上以0.02为间隔，生成网格点
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])#预测每个网格点的类别0/1
    Z = Z.reshape(xx.shape) #转型为网格的形状
    plt.contour(xx, yy,Z, levels=[0],colors='r')  #等高线图 将0/1分界线（决策边界）画出来