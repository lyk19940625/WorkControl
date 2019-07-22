
import numpy as np
import cv2

def line_detect_possible_demo(image):
    global img
    # 提取图片中黄色区域(hsv空间)
    lower_blue = np.array([26, 43, 46])
    upper_blue = np.array([34, 255, 255])
    frame = image
    # cv2.imshow('Capture', frame)
    # change to hsv model
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # get mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # detect red
    res = cv2.bitwise_and(frame, frame, mask=mask)
    ret, binary = cv2.threshold(res, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    # 图像进行膨胀处理
    dilation = cv2.dilate(binary, kernel)
    # 图像腐蚀处理
    erosion = cv2.erode(dilation, kernel)
    erosion = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(erosion, 100, 150, apertureSize=3)
    #edges = cv2.Canny(binary, 50, 150, apertureSize=3)  # apertureSize是sobel算子大小，只能为1,3,5，7
    # cv2.imshow('1',edges)
    # cv2.waitKey(0)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=0,maxLineGap=10)  #函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    list= []
    for i in range(len(lines)):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        list.append((x1,y1,x2,y2))
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    return list
def cross(p1,p2,p3):#跨立实验
    x1=p2[0]-p1[0]
    y1=p2[1]-p1[1]
    x2=p3[0]-p1[0]
    y2=p3[1]-p1[1]
    return x1*y2-x2*y1

def IsIntersec(p1,p2,p3,p4): #判断两线段是否相交
    #快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if(max(p1[0],p2[0])>=min(p3[0],p4[0])    #矩形1最右端大于矩形2最左端
    and max(p3[0],p4[0])>=min(p1[0],p2[0])   #矩形2最右端大于矩形最左端
    and max(p1[1],p2[1])>=min(p3[1],p4[1])   #矩形1最高端大于矩形最低端
    and max(p3[1],p4[1])>=min(p1[1],p2[1])): #矩形2最高端大于矩形最低端

    #若通过快速排斥则进行跨立实验
        if(cross(p1,p2,p3)*cross(p1,p2,p4)<=0
           and cross(p3,p4,p1)*cross(p3,p4,p2)<=0):
            D='有交点'
        else:
            D='无交点'
    else:
        D='无交点'
    return D

if __name__ == "__main__":
    # 获取警戒线
    cap = cv2.VideoCapture("rtsp://admin:IVDCRX@192.168.1.139:554//Streaming/Channels/1")
    ret, frame = cap.read()
    cv2.imwrite('111.jpg',frame)
    img = cv2.imread('/home/user/PycharmProjects/tracking/unofficial-openpose/111.jpg')
    cv2.line(img, (275,360), (378, 360), (0, 255, 255), 1)
    transboundaryline = line_detect_possible_demo(img)
    cv2.imshow('1', img)
    cv2.waitKey(0)