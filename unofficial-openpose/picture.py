import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def picture():
    #定义图像和三维格式坐标轴
    fig=plt.figure()
    ax1 = Axes3D(fig)
    z = np.linspace(0,13,1000)
    x = 5*np.sin(z)
    y = 5*np.cos(z)
    ax1.plot3D(x,y,z,'gray')    #绘制空间曲线
    plt.show()
picture()