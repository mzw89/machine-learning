import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    plt.figure()

    positive=X[y==1] #提取正样本
    negtive=X[y==0]  #提取负样本
    
    plt.scatter(positive[:,0],positive[:,1],marker='+',label='y=1') #画出正样本
    plt.scatter(negtive[:,0],negtive[:,1],marker='o',label='y=0') #画出负样本
    plt.legend() #显示图例



