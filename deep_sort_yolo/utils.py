import cv2
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
cps = []
KalmanNmae = {}
workers = {}
lmp = {}
class KalmanFilters:

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                      np.float32) * 0.03  # 系统过程噪声协方差

    def Estimate(self, coordX, coordY):
        ''' 此函数估计对象的位置'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        print('____________________________________________________________')
        print("measured",measured)
        print("predicted",predicted)
        return predicted

def myKalman(tid):
    if tid not in workers:
        kalman = cv2.KalmanFilter(4, 2) # 4：状态数，包括（x，y，dx，dy）坐标及速度（每次移动的距离）；2：观测量，能看到的是坐标值
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) # 系统测量矩阵
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) # 状态转移矩阵
        kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)*0.03 # 系统过程噪声协方差
        KalmanNmae[tid] = kalman

def setLMP(tid):
    last = []
    last_measurement = current_measurement = np.array((2, 1), np.float32)
    last_prediction = current_prediction = np.zeros((2, 1), np.float32)
    last.append(last_measurement)
    last.append(last_prediction)
    last.append(current_measurement)
    last.append(current_prediction)
    lmp[tid] = last
def predict(x, y,tid):
    lmp[tid][1] = lmp[tid][3] # 把当前预测存储为上一次预测
    lmp[tid][0] = lmp[tid][2] # 把当前测量存储为上一次测量
    lmp[tid][2] = np.array([[np.float32(x)], [np.float32(y)]]) # 当前测量
    KalmanNmae[tid].correct(lmp[tid][2]) # 用当前测量来校正卡尔曼滤波器
    lmp[tid][3] = KalmanNmae[tid].predict() # 计算卡尔曼预测值，作为当前预测
    cpx, cpy = lmp[tid][3][0], lmp[tid][3][1] # 当前预测坐标
    return(cpx,cpy)

def draw(frame,line):

    for i in range(len(line)-1):

        p1 = (int(line[i][0]),int(line[i][1]))
        p2 = (int(line[i+1][0]),int(line[i+1][1]))
        cv2.line(frame, p1, p2, (0, 255, 255), 3)


def line(pointl, pointr):
    # 处理表格
    df = pd.read_excel('grid.xlsx', index_col=False)
    input2DX = []
    input2DY = []
    output2DX = []
    output2DY = []
    inputXY = []
    outputXYZ = []
    dictf1 = []
    dictf2 = []
    df1 = {}
    df2 = {}
    ouf1 = []
    ouf2 = []
    for i in df:
        if i == 'inputX':
            input2DX.append(df[i])
        elif i == 'inputY':
            input2DY.append(df[i])
        elif i == 'outputX':
            output2DX.append(df[i])
        else:
            output2DY.append(df[i])
    for i in range(len(input2DX[0])):
        inputXY.append((int(input2DX[0][i]), int(input2DY[0][i])))
        outputXYZ.append((output2DX[0][i], output2DY[0][i]))
    # 创建像素点到三维坐标的映射关系（字典形式）
    dict2to3 = dict(zip(inputXY, outputXYZ))
    for j in range(len(inputXY)):
        Ous = np.sqrt((inputXY[j][0] - pointl[0][0]) ** 2 + (inputXY[j][1] - pointl[0][1]) ** 2)
        ouf1.append(Ous)
    for j in range(len(inputXY)):
        Ous = np.sqrt((inputXY[j][0] - pointr[0][0]) ** 2 + (inputXY[j][1] - pointr[0][1]) ** 2)
        ouf2.append(Ous)
    df1[pointl[0]] = dict(zip(inputXY, ouf1))
    df1[pointl[0]] = sorted(df1[pointl[0]].items(), key=lambda x: x[1], reverse=False)
    df2[pointr[0]] = dict(zip(inputXY, ouf2))
    df2[pointr[0]] = sorted(df2[pointr[0]].items(), key=lambda x: x[1], reverse=False)
    # print(sorted(d[point[i]].items(),key = lambda x:x[1],rreverse = False))
    for i in pointl:
        s = dict2to3[df1[i][0][0]]
        dictf1.append(s)
    for i in pointr:
        s = dict2to3[df2[i][0][0]]
        dictf2.append(s)
    print(dictf1, dictf2)
    z0 = 2
    f1 = (dictf1[0][0], dictf1[0][1], 0)
    f2 = (dictf2[0][0], dictf2[0][1], 0)
    h = ((dictf1[0][0] + dictf2[0][0]) / 2, (dictf1[0][1] + dictf2[0][1]) / 2, z0)
    zh = ((dictf1[0][0] + dictf2[0][0]) / 2, (dictf1[0][1] + dictf2[0][1]) / 2, z0 / 2.5)
    zb = ((dictf1[0][0] + dictf2[0][0]) / 2, (dictf1[0][1] + dictf2[0][1]) / 2, z0 / 1.6)
    hl = (dictf2[0][0], dictf2[0][1], z0 / 2)
    hr = (dictf1[0][0], dictf1[0][1], z0 / 2)
    x = [h[0], zh[0]]
    y = [h[1], zh[1]]
    z = [z0, z0 / 2.5]
    x1 = [f1[0], zh[0]]
    y1 = [f1[1], zh[1]]
    z1 = [0, z0 / 2.5]
    x2 = [f2[0], zh[0]]
    y2 = [f2[1], zh[1]]
    z2 = [0, z0 / 2.5]
    x3 = [hl[0], zb[0]]
    y3 = [hl[1], zb[1]]
    z3 = [z0 / 2, z0 / 1.6]
    x4 = [hr[0], zb[0]]
    y4 = [hr[1], zb[1]]
    z4 = [z0 / 2, z0 / 1.6]
    mpl.rcParams['legend.fontsize'] = 10
    ax = plt.axes(projection='3d')
    ax.set_xlim([0, 7])
    ax.set_ylim([0, 12])
    ax.set_zlim([0, 5])
    ax.plot(x, y, z, c='r')
    ax.plot(x1, y1, z1, c='r')
    ax.plot(x2, y2, z2, c='r')
    ax.plot(x3, y3, z3, c='r')
    ax.plot(x4, y4, z4, c='r')
    plt.pause(0.1)
    print('f1,f2,h,z', f1, f2, h, zh)
    return f1, f2, h, zh, zb, hl, hr

