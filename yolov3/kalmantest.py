import cv2
import numpy as np


# 初始化测量坐标和鼠标运动预测的数组
last_measurement = current_measurement = np.array((2, 1), np.float32)
last_prediction = current_prediction = np.zeros((2, 1), np.float32)
cps = []
# 定义鼠标回调函数，用来绘制跟踪结果
def predict(x, y):
    global  current_measurement, last_measurement, current_prediction, last_prediction
    last_prediction = current_prediction # 把当前预测存储为上一次预测
    last_measurement = current_measurement # 把当前测量存储为上一次测量
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]]) # 当前测量
    kalman.correct(current_measurement) # 用当前测量来校正卡尔曼滤波器
    current_prediction = kalman.predict() # 计算卡尔曼预测值，作为当前预测

    lmx, lmy = last_measurement[0], last_measurement[1] # 上一次测量坐标
    cmx, cmy = current_measurement[0], current_measurement[1] # 当前测量坐标
    lpx, lpy = last_prediction[0], last_prediction[1] # 上一次预测坐标
    cpx, cpy = current_prediction[0], current_prediction[1] # 当前预测坐标

    cp = []
    cp.append(cpx)
    cp.append(cpy)
    cps.append(cp)
    print(cpx,cpy)


kalman = cv2.KalmanFilter(4, 2) # 4：状态数，包括（x，y，dx，dy）坐标及速度（每次移动的距离）；2：观测量，能看到的是坐标值
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) # 系统测量矩阵
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) # 状态转移矩阵
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)*0.03 # 系统过程噪声协方差

def getXY(filename):
    import xlrd
    data = xlrd.open_workbook(filename)  # 打开excel
    table = data.sheet_by_name("test")  # 读sheet
    nrows = table.nrows  # 获得行数

    xy = []
    for i in range(1, nrows):  #
        rows = table.row_values(i)  # 行的数据放在数组里:
        if rows[0].strip()!='':
            result = []
            result.append(float(rows[0].split(',')[1]))
            result.append(float(rows[0].split(',')[2]))
            xy.append(result)
            
    print(xy)
    return xy

xy = getXY('myData.xls')

for data in xy:
    x = data[0]
    y = data[1]
    predict(x,y)

import matplotlib.pyplot as plt
import numpy as np

myXY = np.array(xy)

px = myXY[:,0]
py = myXY[:,1]
plt.plot(px, py, label='predict', color='black')
myT = np.array(cps)
cx = myT[:,0]
cy = myT[:,1]
plt.plot(cx, cy, label='true', color='red')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Data')
plt.legend()
plt.show()
