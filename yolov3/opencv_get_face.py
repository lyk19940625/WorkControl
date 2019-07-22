import cv2
from collections import deque
import numpy as np


# 改变亮度与对比度
def relight(img, alpha=2, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    #image = []
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp = int(img[j,i,c]*alpha + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img

def CatchPICFromVideo(window_name, camera_idx):
    cv2.namedWindow(window_name)

    # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)

    # 告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier("G:/PythonWorkSpace/kalman/haarcascade_frontalface_alt2.xml")
    pts =  deque(maxlen=60)
    # 识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)

    num = 0
    while cap.isOpened():
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像

        # 人脸检测，1.3和5分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=5, minSize=(32, 32))
        l = len(faceRects)
        print(l)
        for (x,y,w,h) in faceRects:
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,'face',(w + x,y-h),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2,1)
            center = (int(x+w/2),int(y+h))
            print(center)
            pts.appendleft(center)
            pt = []
            for i in range(1,len(pts)):
                if pts[i-1]is None or pts[i]is None:
                    continue
                thickness = int(np.sqrt(64/float(i+1))*2.5)
                # pts = list(pts)
                # for j in pts:
                #     pt.append(int(j))
                # pts = tuple(pt)
                print(pts[i-1],pts[i])
                cv2.line(frame,pts[i-1],pts[i],(0,0,255),thickness)
            cv2.imshow(window_name, frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break

            # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()



CatchPICFromVideo("face", 0)