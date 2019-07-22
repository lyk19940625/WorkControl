#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import sys
sys.path.append('../')
import math
from timeit import time
import warnings
import numpy as np
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from yolov3.yolo2 import YOLO as YOLO2
#import pymysql
import utils
import operator
import json
import line
import queue
import os
import cv2
import gc
from multiprocessing import Process, Manager
#conn=pymysql.connect(host='120.24.15.17',user='root',passwd='123456',db='track',port=3306,charset='gbk')
info_queue = queue.Queue()
warnings# 向共享缓冲栈中写入数据:
def write(stack, cam, top) :
    """
    :param cam: 摄像头参数
    :param stack: Manager.list对象
    :param top: 缓冲栈容量
    :return: None
    """
    print('Process to write: %s' % os.getpid())
    cap = cv2.VideoCapture(cam)
    while True:
        _, img = cap.read()
        if _:
            stack.append(img)
            # 每到一定容量清空一次缓冲栈
            # 利用gc库，手动清理内存垃圾，防止内存溢出
            if len(stack) >= top:
                del stack[:]
                gc.collect()

def read(stack) :
    print('Process to read: %s' % os.getpid())
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
    max_boxs = 0
    face = ['A']
    # face = []
    # cur1 = conn.cursor()  # 获取一个游标
    # sql = "select * from worker"
    # cur1.execute(sql)
    # data = cur1.fetchall()
    # for d in data:
    #     # 注意int类型需要使用str函数转义
    #     name = str(d[1])
    #
    #     face.append(name)
    # cur1.close()  # 关闭游标
    yolo2 = YOLO2()
    #目标上一帧的点
    history = {}
    #id和标签的字典
    person = {}
    #赋予新标签的id列表
    change = []
    while True:
        if len(stack) != 0:
            frame = stack.pop()
            t1 = time.time()
            localtime = time.asctime(time.localtime(time.time()))
            # 进行安全措施检测
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = Image.fromarray(frame)
            #img.save('frame.jpg')
            frame, wear = yolo2.detect_image(img)
            frame = np.array(frame)
           # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 获取警戒线
            transboundaryline = line.readline()
            utils.draw(frame, transboundaryline)
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
            info = {}
            target = []
            for track in tracker.tracks:
                #一帧中的目标
                per_info = {}
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                if track.track_id not in person:
                    person[track.track_id] = str(track.track_id)
                bbox = track.to_tlbr()
                PointX = bbox[0] + ((bbox[2] - bbox[0]) / 2)
                PointY = bbox[3]
                dis = int(PointX) - 1200
                try:
                    if dis<15:
                        if track.track_id not in change:
                            person[track.track_id] = face.pop(0)
                            change.append(track.track_id)
                except:
                    print('非法入侵')
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
                    if wear_dic[0][0] < 180:
                        if wear[wear_dic[0][1]] == 1:
                            per_info['wear'] = 'no_helmet'

                        elif wear[wear_dic[0][1]] == 2:
                            per_info['wear'] = 'no work cloths'

                        elif wear[wear_dic[0][1]] == 3:
                            per_info['wear'] = 'unsafe wear'
                # 写入越线情况
                if per_info['worker_id'] in history:
                    per_info['transboundary'] = 'no'
                    # print(transboundaryline)

                    line1 = [per_info['next_point'], history[per_info['worker_id']]]
                    a = line.IsIntersec2(transboundaryline, line1)

                    if a == '有交点':
                        print('越线提醒')

                        per_info['transboundary'] = 'yes'

                history[per_info['worker_id']] = per_info['current_point']

                #print(per_info)
                #画目标框
                #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, per_info['worker_id'], (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
                target.append(per_info)
            info['time'] = localtime
            #info['frame'] = str(img.tolist()).encode('base64')
            info['frame'] = 'frame'
            info['target'] = target
            #写入josn
            info_json = json.dumps(info)
            info_queue.put(info_json)
            getInfo(info_queue)
            cv2.imshow("img", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break


def getInfo(queue):
    a = queue.get()
    print(a)

if __name__ == '__main__':
    # 父进程创建缓冲栈，并传给各个子进程：
    q = Manager().list()
    #pw = Process(target=write, args=(q, "rtsp://admin:yfzx2019@192.168.10.6:554/h264/ch1/main/av_stream", 100))
    pw = Process(target=write, args=(q, "rtsp://admin:WFGMYS@192.168.1.107:554//h264/ch1/main/Channels/1", 100))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()

    # 等待pr结束:
    pr.join()

    # pw进程里是死循环，无法等待其结束，只能强行终止:
    pw.terminate()