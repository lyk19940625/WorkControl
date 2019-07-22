#写入数据库2.0
from __future__ import division, print_function, absolute_import
import queue
import threading
from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import math
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import copy
import operator
import worker as wk
from yolov3.yolo2 import YOLO as YOLO2
import pymysql
import utils
import transboundary as t
conn=pymysql.connect(host='120.24.15.17',user='root',passwd='123456',db='track',port=3306,charset='gbk')
warnings.filterwarnings('ignore')
workers = {}
q = queue.Queue()



def Receive():
    print("start Reveive")
    cap = cv2.VideoCapture("deep2.mp4")
    #cap = cv2.VideoCapture("rtsp://admin:IVDCRX@192.168.1.139:554//Streaming/Channels/1")
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)


def Display():
    print("Start Displaying")
    yolo = YOLO()
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    writeVideo_flag = True
    w = 768
    h = 432
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
    list_file = open('detection.txt', 'w')
    frame_index = -1

    fps = 0.0
    max_boxs = 0
    person_track = {}
    yolo2 = YOLO2()
    while True:
        if q.empty() != True:
            #读取打卡信息
            face = []
            cur1 = conn.cursor()  # 获取一个游标
            sql1 = "select * from worker"
            cur1.execute(sql1)
            data = cur1.fetchall()
            for d in data:
                # 注意int类型需要使用str函数转义
                name = str(d[1]) + '_' + d[2]
                face.append(name)
            cur1.close()  # 关闭游标
            #获取队列帧
            frame = q.get()
            t1 = time.time()

            #进行安全措施检测
            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame, wear = yolo2.detect_image(img)
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 获取警戒线
            #cv2.line(frame, (132,368), (229, 368), (0, 255, 255), 3)
            cv2.line(frame, (275,360), (378, 360), (0, 255, 255), 1)
            transboundaryline = t.line_detect_possible_demo(frame)
            #yolo目标检测
            boxs = yolo.detect_image(image)
            features = encoder(frame, boxs)
            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            if len(boxs) > max_boxs:
                max_boxs = len(boxs)

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if max_boxs < track.track_id:
                    tracker.tracks.remove(track)
                    tracker._next_id = max_boxs + 1
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                PointX = bbox[0] + ((bbox[2] - bbox[0]) / 2)
                PointY = bbox[3]

                if track.track_id not in person_track:
                    track2 = copy.deepcopy(track)
                    person_track[track.track_id] = track2

                else:
                    track2 = copy.deepcopy(track)
                    bbox2 = person_track[track.track_id].to_tlbr()
                    PointX2 = bbox2[0] + ((bbox2[2] - bbox2[0]) / 2)
                    PointY2 = bbox2[3]
                    distance = math.sqrt(pow(PointX - PointX2, 2) + pow(PointY - PointY2, 2))
                    #修正
                    if distance < 120:
                        person_track[track.track_id] = track2
                    else:
                        # print('last',track.track_id)
                        dis = {}
                        for key in person_track:
                            bbox3 = person_track[key].to_tlbr()
                            PointX3 = bbox3[0] + ((bbox3[2] - bbox3[0]) / 2)
                            PointY3 = bbox3[3]
                            d = math.sqrt(pow(PointX3 - PointX, 2) + pow(PointY3 - PointY, 2))
                            dis[key] = d
                        dis = sorted(dis.items(), key=operator.itemgetter(1), reverse=False)
                        track2.track_id = dis[0][0]
                        person_track[dis[0][0]] = track2
                        tracker.tracks.remove(track)
                        tracker.tracks.append(person_track[track.track_id])

                # 写入class

                try:
                    box_title = face[track2.track_id - 1]
                except Exception as e:
                    box_title = str(track2.track_id) + "_" + "unknow"
                if box_title not in workers:
                    wid = box_title.split('_')[0]
                    localtime = time.asctime(time.localtime(time.time()))
                    workers[box_title] = wk.Worker()
                    workers[box_title].set(box_title, localtime, (int(PointX), int(PointY)))
                    cur2 = conn.cursor()  # 获取一个游标
                    sql2 = "UPDATE worker SET in_time='" + localtime + "' WHERE worker_id= '" + wid + "'"
                    cur2.execute(sql2)
                    cur2.close()  # 关闭游标
                else:
                    localtime = time.asctime(time.localtime(time.time()))
                    yoloPoint = (int(PointX), int(PointY))
                    wear_dic = {}
                    workers[box_title].current_point = yoloPoint
                    workers[box_title].track_point.append(workers[box_title].current_point)
                    mytrack = str(workers[box_title].track_point)
                    wid = box_title.split('_')[0]
                    # 卡尔曼滤波预测
                    if wid not in utils.KalmanNmae:
                        utils.myKalman(wid)
                    if wid not in utils.lmp:
                        utils.setLMP(wid)
                    cpx, cpy = utils.predict(workers[box_title].current_point[0], workers[box_title].current_point[1], wid)

                    if cpx[0] == 0.0 or cpy[0] == 0.0:
                        cpx[0] = workers[box_title].current_point[0]
                        cpy[0] = workers[box_title].current_point[1]
                    workers[box_title].next_point = (int(cpx), int(cpy))

                    cur3 = conn.cursor()  # 获取一个游标
                    sql3 = "UPDATE worker SET current_point= '" + str(workers[box_title].current_point) + "' ,track_point = '" + mytrack + "',next_point = '" + str(workers[box_title].next_point) + "' WHERE worker_id= '" + wid + "'"
                    cur3.execute(sql3)
                    cur3.close()
                    # 写入安全措施情况
                    if len(wear) > 0:
                        for w in wear:
                            wear_dis = int(math.sqrt(pow(w[0] - yoloPoint[0], 2) + pow(w[1] - yoloPoint[1], 2)))
                            wear_dic[wear_dis] = w
                        wear_dic = sorted(wear_dic.items(), key=operator.itemgetter(0), reverse=False)

                        if wear_dic[0][0] < 120:
                            cur4 = conn.cursor()  # 获取一个游标

                            if wear[wear_dic[0][1]] == 1:
                                if len(workers[box_title].wear['no helmet']) == 0:

                                    workers[box_title].wear['no helmet'].append(localtime)
                                    sql = "INSERT INTO wear SET worker_id = '" + wid + "', type = 'no_helmet',abnormal_time = '" + localtime + "'"
                                    cur4.execute(sql)
                                    cur4.close()  # 关闭游标

                                else:
                                    print(box_title,workers[box_title].wear['no helmet'])
                                    if localtime not in workers[box_title].wear['no helmet']:
                                        workers[box_title].wear['no helmet'].append(localtime)
                                        sql = "INSERT INTO wear SET worker_id = '" + wid + "', type = 'no_helmet',abnormal_time = '" + localtime + "'"
                                        cur4.execute(sql)
                                        cur4.close()  # 关闭游标


                            elif wear[wear_dic[0][1]] == 2:
                                if len(workers[box_title].wear['no work cloths']) == 0:
                                    workers[box_title].wear['no work cloths'].append(localtime)
                                    sql = "INSERT INTO wear SET worker_id = '" + wid + "', type = 'no work cloths',abnormal_time = '" + localtime + "'"
                                    cur4.execute(sql)
                                    cur4.close()  # 关闭游标
                                else:
                                    if localtime not in workers[box_title].wear['no work cloths']:
                                        workers[box_title].wear['no work cloths'].append(localtime)
                                        sql = "INSERT INTO wear SET worker_id = '" + wid + "', type = 'no work cloths',abnormal_time = '" + localtime + "'"
                                        cur4.execute(sql)
                                        cur4.close()  # 关闭游标
                            elif wear[wear_dic[0][1]] == 3:
                                if len(workers[box_title].wear['unsafe wear']) == 0:
                                    workers[box_title].wear['unsafe wear'].append(localtime)
                                    sql = "INSERT INTO wear SET worker_id = '" + wid + "', type = 'unsafe wear',abnormal_time = '" + localtime + "'"
                                    cur4.execute(sql)
                                    cur4.close()  # 关闭游标
                                else:
                                    if localtime not in workers[box_title].wear['unsafe wear']:
                                        workers[box_title].wear['unsafe wear'].append(localtime)
                                        sql = "INSERT INTO wear SET worker_id = '" + wid + "', type = 'unsafe wear',abnormal_time = '" + localtime + "'"
                                        cur4.execute(sql)
                                        cur4.close()  # 关闭游标

                    # 写入越线情况
                    if len(workers[box_title].track_point) > 4:

                        for i in range(len(transboundaryline)):
                            p1 = (transboundaryline[i][0], transboundaryline[i][1])
                            p2 = (transboundaryline[i][2], transboundaryline[i][3])
                            p3 = workers[box_title].track_point[-2]
                            p4 = workers[box_title].track_point[-1]
                            a = t.IsIntersec(p1, p2, p3, p4)
                            if a == '有交点':
                                cur5 = conn.cursor()  # 获取一个游标
                                cur6 = conn.cursor()  # 获取一个游标
                                cur5.execute(
                                    "select time from transboundary where worker_id = '" + wid + "' ")

                                qurrytime = cur5.fetchone()
                                cur5.close()  # 关闭游标
                                if qurrytime == None:
                                    print('越线')
                                    sql = "INSERT INTO transboundary SET worker_id = '" + wid + "',time = '" + localtime + "'"
                                    cur6.execute(sql)
                                    cur6.close()  # 关闭游标
                                else:
                                    temp1 = 0
                                    for qt in qurrytime:
                                        if qt == localtime:
                                            temp1 = 1
                                    if temp1 == 0:
                                        print('越线')
                                        sql = "INSERT INTO transboundary SET worker_id = '" + wid + "',time = '" + localtime + "'"
                                        cur6.execute(sql)
                                        cur6.close()  # 关闭游标
                    if len(workers[box_title].track_point) >= 20:
                        workers[box_title].previous_point = workers[box_title].track_point[-5]
                conn.commit()
                try:
                    cv2.putText(frame, face[track2.track_id - 1], (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0),
                                2)
                except Exception as e:
                    cv2.putText(frame, "unknow", (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200,
                                (0, 255, 0), 2)

            cv2.imshow('', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(boxs) != 0:
                for i in range(0, len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(
                        boxs[i][3]) + ' ')
            list_file.write('\n')

        fps = (fps + (1. / (time.time() - t1))) / 2
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()