import cv2


#读取视频
cap = cv2.VideoCapture("rtsp://admin:IVDCRX@192.168.1.139:554//Streaming/Channels/1")  # 打开摄像头
ret,frame = cap.read()
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi', fourcc, 6.0, size)

while (1):
    ret, frame = cap.read()
    if ret == True:
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()



'''
""" 从视频读取帧保存为图片"""
import cv2

cap = cv2.VideoCapture("output1.avi")
print(cap.isOpened())
frame_count = 1
success = True
while (success):
    success, frame = cap.read()
    print('Read a new frame: ', success)
    params = []
    # params.append(cv.CV_IMWRITE_PXM_BINARY)
    params.append(1)
    cv2.imwrite("_%d.jpg" % frame_count, frame, params)

    frame_count = frame_count + 1

cap.release()
'''


