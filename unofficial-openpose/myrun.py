from __future__ import division, print_function, absolute_import
import argparse
import tensorflow as tf
import copy
import logging
from tensorflow.contrib import slim
import vgg
from cpm import PafNet
import common
from tensblur.smoother import Smoother
from estimator import PoseEstimator, TfPoseEstimator
import warnings
import sys
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import operator
import numpy as np
import worker as wk
from yolov3.yolo2 import YOLO as YOLO2
import math
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import time
import pymysql
import transboundary as t
conn=pymysql.connect(host='120.24.15.17',user='root',passwd='123456',db='track',port=3306,charset='gbk')
warnings.filterwarnings('ignore')

logger = logging.getLogger('run')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
currentDT = time.localtime()
start_datetime = time.strftime("-%m-%d-%H-%M-%S", currentDT)



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

cps = []
KalmanNmae = {}
workers = {}
def myKalman(tid):
    if tid not in workers:
        kalman = cv2.KalmanFilter(4, 2) # 4：状态数，包括（x，y，dx，dy）坐标及速度（每次移动的距离）；2：观测量，能看到的是坐标值
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) # 系统测量矩阵
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) # 状态转移矩阵
        kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)*0.03 # 系统过程噪声协方差
        KalmanNmae[tid] = kalman
lmp = {}
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

def line(pointl,pointr):
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
def main():

    yolo = YOLO()
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/train/2018-12-13-16-56-49/')
    parser.add_argument('--backbone_net_ckpt_path', type=str, default='checkpoints/vgg/vgg_19.ckpt')
    parser.add_argument('--image', type=str, default=None)
    # parser.add_argument('--run_model', type=str, default='img')
    parser.add_argument('--video', type=str, default= None)
    parser.add_argument('--train_vgg', type=bool, default=True)
    parser.add_argument('--use_bn', type=bool, default=False)
    parser.add_argument('--save_video', type=str, default='result/our.mp4')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    logger.info('checkpoint_path: ' + checkpoint_path)

    with tf.name_scope('inputs'):
        raw_img = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        img_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='original_image_size')

    img_normalized = raw_img / 255 - 0.5

    # define vgg19
    with slim.arg_scope(vgg.vgg_arg_scope()):
        vgg_outputs, end_points = vgg.vgg_19(img_normalized)

    # get net graph
    logger.info('initializing model...')
    net = PafNet(inputs_x=vgg_outputs, use_bn=args.use_bn)
    hm_pre, cpm_pre, added_layers_out = net.gen_net()
    hm_up = tf.image.resize_area(hm_pre[5], img_size)
    cpm_up = tf.image.resize_area(cpm_pre[5], img_size)
    # hm_up = hm_pre[5]
    # cpm_up = cpm_pre[5]
    smoother = Smoother({'data': hm_up}, 25, 3.0)
    gaussian_heatMat = smoother.get_output()

    max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
    tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat,
                                 tf.zeros_like(gaussian_heatMat))

    logger.info('initialize saver...')
    # trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='openpose_layers')
    # trainable_var_list = []
    trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='openpose_layers')
    if args.train_vgg:
        trainable_var_list = trainable_var_list + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19')

    restorer = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19'), name='vgg_restorer')
    saver = tf.train.Saver(trainable_var_list)

    logger.info('initialize session...')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.group(tf.global_variables_initializer()))
        logger.info('restoring vgg weights...')
        restorer.restore(sess, args.backbone_net_ckpt_path)
        logger.info('restoring from checkpoint...')
        #saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path))
        saver.restore(sess, args.checkpoint_path + 'model-59000.ckpt')
        logger.info('initialization done')
        writeVideo_flag = True
        if args.image is None:
            if args.video is not None:
                cap = cv2.VideoCapture(args.video)
                w = int(cap.get(3))
                h = int(cap.get(4))

            else:
                cap = cv2.VideoCapture("images/video.mp4")
                #cap = cv2.VideoCapture("rtsp://admin:IKVVSA@192.168.43.51:554//Streaming/Channels/1")
                #cap = cv2.VideoCapture("http://admin:admin@192.168.1.111:8081")
                #cap = cv2.VideoCapture("rtsp://admin:IVDCRX@192.168.1.106:554//Streaming/Channels/1")
            _, image = cap.read()
            #print(_,image)
            if image is None:
                logger.error("Can't read video")
                sys.exit(-1)
            fps = cap.get(cv2.CAP_PROP_FPS)
            ori_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ori_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #print(fps,ori_w,ori_h)
            if args.save_video is not None:
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                video_saver = cv2.VideoWriter('result/our.mp4', fourcc, fps, (ori_w, ori_h))
                logger.info('record vide to %s' % args.save_video)
            logger.info('fps@%f' % fps)
            size = [int(654 * (ori_h / ori_w)), 654]
            h = int(654 * (ori_h / ori_w))
            time_n = time.time()
            #print(time_n)

            max_boxs = 0
            person_track = {}
            yolo2 = YOLO2()

            while True:
                face = []
                cur1 = conn.cursor()  # 获取一个游标
                sql = "select * from worker"
                cur1.execute(sql)
                data = cur1.fetchall()
                for d in data:
                    # 注意int类型需要使用str函数转义
                    name = str(d[1]) + '_' + d[2]

                    face.append(name)
                cur1.close()  # 关闭游标

                _, image_fist = cap.read()
                #穿戴安全措施情况检测

                img = Image.fromarray(cv2.cvtColor(image_fist, cv2.COLOR_BGR2RGB))
                image,wear = yolo2.detect_image(img)
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # # 获取警戒线
                cv2.line(image, (837, 393), (930, 300), (0, 255, 255), 3)
                transboundaryline = t.line_detect_possible_demo(image)

                #openpose二维姿态检测
                img = np.array(cv2.resize(image, (654, h)))
                # cv2.imshow('raw', img)
                img_corner = np.array(cv2.resize(image, (360, int(360 * (ori_h / ori_w)))))
                img = img[np.newaxis, :]
                peaks, heatmap, vectormap = sess.run([tensor_peaks, hm_up, cpm_up],
                                                     feed_dict={raw_img: img, img_size: size})
                bodys = PoseEstimator.estimate_paf(peaks[0], heatmap[0], vectormap[0])

                image, person = TfPoseEstimator.draw_humans(image, bodys, imgcopy=False)
                #取10右脚 13左脚

                foot = []
                if len(person) > 0:
                    for p in person:
                        foot_lr = []
                        if 10 in p and 13 in p:
                            foot_lr.append(p[10])
                            foot_lr.append(p[13])

                        if len(foot_lr) > 1:
                            foot.append(foot_lr)

                fps = round(1 / (time.time() - time_n), 2)
                image = cv2.putText(image, str(fps) + 'fps', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (255, 255, 255))
                time_n = time.time()

                #deep目标检测
                image2 = Image.fromarray(image_fist)
                boxs = yolo.detect_image(image2)
                features = encoder(image, boxs)
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
                boxes = np.array([d.tlwh for d in detections])
                scores = np.array([d.confidence for d in detections])
                indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                detections = [detections[i] for i in indices]
                if len(boxs) > max_boxs:
                    max_boxs = len(boxs)
                # print(max_boxs)

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
                        box_title = str(track2.track_id)+"_"+"unknow"
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
                        foot_dic = {}
                        wear_dic = {}

                        for f in foot:
                            fp = []
                            footCenter = ((f[0][0]+f[1][0])/2,(f[0][1]+f[1][1])/2)
                            foot_dis = int(math.sqrt(pow(footCenter[0] - yoloPoint[0], 2) + pow(footCenter[1] - yoloPoint[1], 2)))
                            #print(foot_dis)
                            fp.append(f)
                            fp.append(footCenter)
                            foot_dic[foot_dis] = fp

                        #print(box_title, 'sss', foot_dic)
                        foot_dic = sorted(foot_dic.items(), key=operator.itemgetter(0), reverse=False)
                        workers[box_title].current_point = foot_dic[0][1][1]
                        workers[box_title].track_point.append(workers[box_title].current_point)

                        #print(box_title,'sss',foot_dic[0][1][1])
                        mytrack = str(workers[box_title].track_point)
                        wid = box_title.split('_')[0]
                        #卡尔曼滤波预测
                        if wid not in KalmanNmae:
                            myKalman(wid)
                        if wid not in lmp:
                            setLMP(wid)
                        cpx, cpy = predict(workers[box_title].current_point[0], workers[box_title].current_point[1], wid)

                        if cpx[0] == 0.0 or cpy[0] == 0.0:
                            cpx[0] = workers[box_title].current_point[0]
                            cpy[0] = workers[box_title].current_point[1]
                        workers[box_title].next_point = (int(cpx),int(cpy))

                        workers[box_title].current_footR = foot_dic[0][1][0][0]
                        workers[box_title].current_footL = foot_dic[0][1][0][1]
                        cur3 = conn.cursor()  # 获取一个游标
                        sql = "UPDATE worker SET current_point= '" + str(workers[
                            box_title].current_point) + "' , current_footR = '" + str(workers[
                                  box_title].current_footR) + "',current_footL = '" + str(workers[
                                  box_title].current_footL) + "',track_point = '" + mytrack + "',next_point = '" + str(workers[box_title].next_point) + "' WHERE worker_id= '" + wid + "'"
                        cur3.execute(sql)
                        cur3.close()
                        #写入安全措施情况
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
                                        if localtime not in  workers[box_title].wear['no helmet']:

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
                                        if localtime not in  workers[box_title].wear['no work cloths']:
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
                                        if localtime not in  workers[box_title].wear['unsafe wear']:
                                            workers[box_title].wear['unsafe wear'].append(localtime)
                                            sql = "INSERT INTO wear SET worker_id = '" + wid + "', type = 'unsafe wear',abnormal_time = '" + localtime + "'"
                                            cur4.execute(sql)
                                            cur4.close()  # 关闭游标

                        #写入越线情况

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
                                    cur5.execute("select time from transboundary where worker_id = '" + wid + "' ")
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
                        cv2.putText(image, face[track2.track_id - 1], (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200,(0, 255, 0), 2)
                    except Exception as e:
                        cv2.putText(image, "unknow", (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200,
                                    (0, 255, 0), 2)



                if args.video is not None:
                    image[27:img_corner.shape[0]+27, :img_corner.shape[1]] = img_corner  # [3:-10, :]
                cv2.imshow(' ', image)
                if args.save_video is not None:
                    video_saver.write(image)
                cv2.waitKey(1)
            else:

                image = common.read_imgfile(args.image)
                size = [image.shape[0], image.shape[1]]
                if image is None:
                    logger.error('Image can not be read, path=%s' % args.image)
                    sys.exit(-1)
                h = int(654 * (size[0] / size[1]))
                img = np.array(cv2.resize(image, (654, h)))
                cv2.imshow('ini', img)
                img = img[np.newaxis, :]
                peaks, heatmap, vectormap = sess.run([tensor_peaks, hm_up, cpm_up],
                                                     feed_dict={raw_img: img, img_size: size})
                cv2.imshow('in', vectormap[0, :, :, 0])
                bodys = PoseEstimator.estimate_paf(peaks[0], heatmap[0], vectormap[0])
                image = TfPoseEstimator.draw_humans(image, bodys, imgcopy=False)
                cv2.imshow(' ', image)
                cv2.waitKey(0)




if __name__ == '__main__':


    main()