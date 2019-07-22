
import cv2  # 图像处理的库 OpenCv
import os.path


# OpenCv 调用摄像头
cap = cv2.VideoCapture("rtsp://admin:WFGMYS@192.168.1.110:554//h264/ch1/main/Channels/1")
fps = 5
size = (1280,720)
fourcc = cv2.VideoWriter_fourcc('M','P','4','2')
if os.path.exists(os.getcwd()+'/drawLine.avi'):
    os.remove(os.getcwd()+'/drawLine.avi')
outVideo = cv2.VideoWriter('drawLine.avi',fourcc,fps,size)
state = 0
count = 0
while cap.isOpened():
    # 480 height * 640 width
    flag, img_rd = cap.read()
    kk = cv2.waitKey(1)
    # 待会要写的字体
    font = cv2.FONT_HERSHEY_COMPLEX


    # 添加说明
    cv2.putText(img_rd, "S: Strat", (20, 400), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    if kk == ord('s'):
        print('start')
        state = 1
    if state == 1:
        if count%2 == 0:

            outVideo.write(img_rd)
        count = count + 1

    # 按下 'q' 键退出
    if kk == ord('q'):
        outVideo.release()
        break

    # 窗口显示
    # cv2.namedWindow("camera", 0) # 如果需要摄像头窗口大小可调
    cv2.imshow("camera", img_rd)

# 释放摄像头
cap.release()

# 删除建立的窗口
cv2.destroyAllWindows()
