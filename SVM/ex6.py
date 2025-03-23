
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from sklearn import svm
import plotData as pd
import visualizeBoundary as vb
import gaussianKernel as gk


np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
plt.show()
'''第1部分 加载并可视化数据集'''

print('Loading and Visualizing data ... ')


data = scio.loadmat('ex6data1.mat') #加载矩阵格式的数据集
X = data['X']   #提取原始输入特征矩阵
y = data['y'].flatten() #提取标签 并转换为1维数组
m = y.size #样本数

# 可视化训练集
pd.plot_data(X, y)
plt.show()

input('Program paused. Press ENTER to continue')


'''第2部分 训练SVM(线性核函数) 并可视化决策边界'''

print('Training Linear SVM')

#SVM的代价函数以及训练不用自己写 直接调用程序包
c = 1 #SVM参数 可以通过改变他来观察决策边界的变化
clf = svm.SVC(C=c, kernel='linear', tol=1e-3)  #声明一个线性SVM
clf.fit(X, y)  #训练线性SVM

pd.plot_data(X, y)  #可视化训练集
vb.visualize_boundary(clf, X, 0, 4.5, 1.5, 5) #可视化决策边界
plt.show()
input('Program paused. Press ENTER to continue')



'''第3部分 实现高斯核函数'''

print('Evaluating the Gaussian Kernel')

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2 #高斯核函数参数 
sim = gk.gaussian_kernel(x1, x2, sigma) #利用高斯核函数计算两个向量的相似度

print('Gaussian kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = {} : {:0.6f}\n'
      '(for sigma = 2, this value should be about 0.324652'.format(sigma, sim))

input('Program paused. Press ENTER to continue')


'''第4部分 可视化数据集2'''

print('Loading and Visualizing Data ...')


data = scio.loadmat('ex6data2.mat')#加载矩阵格式的数据集2
X = data['X']  #提取原始输入特征矩阵   2个原始特征
y = data['y'].flatten() #提取标签  转换为1维数组
m = y.size  #样本数

#可视化训练集2
pd.plot_data(X, y)


input('Program paused. Press ENTER to continue')

'''第5部分 对训练集2使用带有高斯核函数的SVM进行训练'''

print('Training SVM with RFB(Gaussian) Kernel (this may take 1 to 2 minutes) ...')

#参数设置
c = 1
sigma = 0.1

#调用自己写的高斯核函数  返回新的特征向量矩阵
def gaussian_kernel(x_1, x_2):
    n1 = x_1.shape[0]
    n2 = x_2.shape[0]
    result = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            result[i, j] = gk.gaussian_kernel(x_1[i], x_2[j], sigma)

    return result

# clf = svm.SVC(c, kernel=gaussian_kernel) #使用自己手写的高斯核函数
clf = svm.SVC(C=c, kernel='rbf', gamma=np.power(sigma, -2)) #使用封装好的高斯核函数 rbf
clf.fit(X, y)  #进行训练

print('Training complete!')

pd.plot_data(X, y) #可视化训练集
vb.visualize_boundary(clf, X, 0, 1, .4, 1.0)  #可视化决策边界
plt.show()
input('Program paused. Press ENTER to continue')

'''第6部分 可视化训练集3'''

print('Loading and Visualizing Data ...')


data = scio.loadmat('ex6data3.mat')#加载矩阵格式的数据集2
X = data['X']
y = data['y'].flatten()
m = y.size

# Plot training data
pd.plot_data(X, y)
plt.show()
input('Program paused. Press ENTER to continue')

# ===================== Part 7: Visualizing Dataset 3 =====================

clf = svm.SVC(C=c, kernel='rbf', gamma=np.power(sigma, -2))
clf.fit(X, y)


pd.plot_data(X, y)
vb.visualize_boundary(clf, X, -.5, .3, -.8, .6)
plt.show()
input('ex6 Finished. Press ENTER to exit')
