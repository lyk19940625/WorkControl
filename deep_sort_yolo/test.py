#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import threading
import queue
import math
from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from yolov3.yolo2 import YOLO as YOLO2
import pymysql
import utils
import operator
import transboundary as t
import json
import line
conn=pymysql.connect(host='120.24.15.17',user='root',passwd='123456',db='track',port=3306,charset='gbk')

warnings.filterwarnings('ignore')
#视频队列
#q=queue.LifoQueue()
q = queue.Queue()
#每帧队列
info_queue = queue.Queue()
def Receive():
    print("start Reveive")

    cap = cv2.VideoCapture("/home/user/PycharmProjects/tracking/media/deep2.mp4")
    ret, frame = cap.read()
    q.put(frame)
    x = 0
    while ret:
        ret, frame = cap.read()
        x = x + 1
        if x%2 == 0:
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
    #face = ['A','B','C']
    face = []
    cur1 = conn.cursor()  # 获取一个游标
    sql1 = "select * from worker"
    cur1.execute(sql1)
    data = cur1.fetchall()
    for d in data:
        # 注意int类型需要使用str函数转义
        name = str(d[1])
        face.append(name)
    cur1.close()  # 关闭游标
    yolo2 = YOLO2()
    #id和标签的字典
    person = {}
    #赋予新标签的id列表
    change = []
    while True:
        if q.empty() != True:
            localtime = time.asctime(time.localtime(time.time()))

            frame = q.get()
            t1 = time.time()
            # 进行安全措施检测

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame, wear = yolo2.detect_image(img)
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 获取警戒线
            cv2.line(frame, (837, 393), (930, 300), (0, 255, 255), 3)
            transboundaryline = t.line_detect_possible_demo(frame)
            # image = Image.fromarray(frame)
            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
            boxs = yolo.detect_image(image)
            # print("box_num",len(boxs))
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
            #一帧信息
            info = []
            for track in tracker.tracks:
                #一帧中的目标
                per_info = {}
                per_info['localtime'] = localtime
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                if track.track_id not in person:
                    person[track.track_id] = str(track.track_id)
                bbox = track.to_tlbr()
                PointX = bbox[0] + ((bbox[2] - bbox[0]) / 2)
                PointY = bbox[3]
                dis = int(PointX) - 1200
                if dis<15:
                    if track.track_id not in change:
                        person[track.track_id] = face.pop(0)
                        change.append(track.track_id)
                #当前目标
                if track.track_id not in change:
                    per_info['worker_id'] = 'unknow'+str(track.track_id)
                else:
                    per_info['worker_id'] = person[track.track_id]
                #当前目标坐标
                yoloPoint = (int(PointX), int(PointY))
                per_info['current_point'] = yoloPoint

                # 卡尔曼滤波预测
                if per_info['worker_id'] not in utils.KalmanNmae:
                    utils.myKalman(per_info['worker_id'])
                if per_info['worker_id'] not in utils.lmp:
                    utils.setLMP(per_info['worker_id'])
                cpx, cpy = utils.predict(yoloPoint[0], yoloPoint[1], per_info['worker_id'])

                if cpx[0] == 0.0 or cpy[0] == 0.0:
                    cpx[0] = yoloPoint[0]
                    cpy[0] = yoloPoint[1]
                per_info['next_point'] = (int(cpx), int(cpy))
                # 写入安全措施情况
                wear_dic = {}
                per_info['wear'] = 'safe wear'
                if len(wear) > 0:
                    for w in wear:
                        wear_dis = int(math.sqrt(pow(w[0] - yoloPoint[0], 2) + pow(w[1] - yoloPoint[1], 2)))
                        wear_dic[wear_dis] = w
                    wear_dic = sorted(wear_dic.items(), key=operator.itemgetter(0), reverse=False)

                    if wear_dic[0][0] < 120:
                        if wear[wear_dic[0][1]] == 1:
                            per_info['wear'] = 'no_helmet'

                        elif wear[wear_dic[0][1]] == 2:
                            per_info['wear'] = 'no work cloths'

                        elif wear[wear_dic[0][1]] == 3:
                            per_info['wear'] = 'unsafe wear'
                # 写入越线情况
                per_info['transboundary'] = 'no'
                #print(transboundaryline)
                for i in range(len(transboundaryline)):
                    track = [[transboundaryline[i][0], transboundaryline[i][1]],[transboundaryline[i][2], transboundaryline[i][3]]]
                    line1 = [per_info['next_point'],per_info['current_point']]
                    a = line.IsIntersec2(track,line1)
                    #a = t.IsIntersec(p1, p2, p3, p4)
                    if a == '有交点':
                        print('越线提醒')
                        per_info['transboundary'] = 'yes'

                #print(per_info)
                #画目标框
                #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, per_info['worker_id'], (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
                info.append(per_info)
            #写入josn
            info_json = json.dumps(info)
            info_queue.put(info_json)
            getInfo(info_queue)

            cv2.imshow('', frame)

            if writeVideo_flag:
                # save a frame
                out.write(frame)
                frame_index = frame_index + 1
                list_file.write(str(frame_index) + ' ')
                if len(boxs) != 0:
                    for i in range(0, len(boxs)):
                        list_file.write(
                            str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
                list_file.write('\n')

            fps = (fps + (1. / (time.time() - t1))) / 2


            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

def getInfo(queue):
    a = queue.get()
    print(a)


if __name__ == '__main__':
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()