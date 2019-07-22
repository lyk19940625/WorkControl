import cv2
cap = cv2.VideoCapture('rtsp://admin:IKVVSA@192.168.1.101:554/h264/ch1/main/av_stream')

ret,frame = cap.read()
print(cap.get(3))
while ret:
    ret,frame = cap.read()
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()