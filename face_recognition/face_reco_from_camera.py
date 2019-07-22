

import dlib          # 人脸处理的库 Dlib
import numpy as np   # 数据处理的库 numpy
import cv2           # 图像处理的库 OpenCv
import pandas as pd  # 数据处理的库 Pandas
import xlrd
import os
import pymysql
class Reco():
    d = os.path.dirname(__file__)  # 返回当前文件所在的目录
    #conn=pymysql.connect(host='120.24.15.17',user='root',passwd='123456',db='track',port=3306,charset='gbk')
    # 人脸识别模型，提取128D的特征矢量
    # face recognition model, the object maps human faces into 128D vectors
    facerec = dlib.face_recognition_model_v1(d+"/data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
    faceList = []

    def __init__(self, ip):
        # 处理存放所有人脸特征的 csv
        path_features_known_csv = self.d+"/data/features_all.csv"
        csv_rd = pd.read_csv(path_features_known_csv, header=None)

        # 存储的特征人脸个数
        # print(csv_rd.shape[0])

        # 用来存放所有录入人脸特征的数组
        features_known_arr = []

        # 读取已知人脸数据
        # known faces
        for i in range(csv_rd.shape[0]):
            features_someone_arr = []
            for j in range(0, len(csv_rd.ix[i, :])):
                features_someone_arr.append(csv_rd.ix[i, :][j])
            features_known_arr.append(features_someone_arr)
        print("Faces in Database：", len(features_known_arr))

        # Dlib 检测器和预测器
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.d+'/data/data_dlib/shape_predictor_68_face_landmarks.dat')

        # 创建 cv2 摄像头对象
        cap = cv2.VideoCapture(ip)
        # cap = cv2.VideoCapture("rtsp://admin:IKVVSA@192.168.1.134:554//Streaming/Channels/1")
        ret, frame = cap.read()
        print(frame.shape)
        # cap = cv2.VideoCapture("http://admin:admin@192.168.1.108:8081")
        # cap.set(propId, value)
        # 设置视频参数，propId 设置的视频参数，value 设置的参数值
        cap.set(3, 480)

        excelName = 'person.xlsx'
        # 读取人员信息excel

        self.personList = []
        path = self.d+'/person.xlsx'
        # cur1 = self.conn.cursor()  # 获取一个游标
        # sql1 = "truncate table worker"
        # sql2 = "truncate table wear"
        # sql3 = "truncate table transboundary"
        # cur1.execute(sql1)
        # cur1.execute(sql2)
        # cur1.execute(sql3)
        # self.conn.commit()
        # cur1.close()  # 关闭游标

        self.get_person(path)

        # cap.isOpened() 返回 true/false 检查初始化是否成功
        while cap.isOpened():

            flag, img_rd = cap.read()
            kk = cv2.waitKey(1)

            # 取灰度
            img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

            # 人脸数 faces
            faces = detector(img_gray, 0)

            # 待会要写的字体
            font = cv2.FONT_HERSHEY_COMPLEX

            # 存储当前摄像头中捕获到的所有人脸的坐标/名字
            pos_namelist = []
            name_namelist = []

            # 按下 q 键退出
            if kk == ord('q'):
                break
            else:
                # 检测到人脸
                if len(faces) != 0:
                    # 获取当前捕获到的图像的所有人脸的特征，存储到 features_cap_arr
                    features_cap_arr = []
                    for i in range(len(faces)):
                        shape = predictor(img_rd, faces[i])
                        features_cap_arr.append(self.facerec.compute_face_descriptor(img_rd, shape))

                    # 遍历捕获到的图像中所有的人脸
                    for k in range(len(faces)):
                        # 让人名跟随在矩形框的下方
                        # 确定人名的位置坐标
                        # 先默认所有人不认识，是 unknown
                        name_namelist.append("unknown")

                        # 每个捕获人脸的名字坐标
                        pos_namelist.append(
                            tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                        # 对于某张人脸，遍历所有存储的人脸特征
                        for i in range(len(features_known_arr)):
                            # print("with person_", str(i+1), "the ", end='')
                            # 将某张人脸与存储的所有人脸数据进行比对
                            compare = self.return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                            if compare == "same":  # 找到了相似脸

                                label1 = str(self.personList[i][1]) + '_' + str(self.personList[i][2])
                                label2 = str(self.personList[i][1]) + '_' + str(self.personList[i][2])
                                name_namelist[k] = label1
                                if name_namelist[k] not in self.faceList:
                                    print(self.personList[i][1])
                                    self.faceList.append(label2)
                                    # cur = self.conn.cursor()  # 获取一个游标
                                    # sql = "INSERT INTO worker SET worker_id = '123', worker_name = 'sss'"
                                    # cur.execute(sql)
                                    # self.conn.commit()
                                    # cur.close()  # 关闭游标
                                    print(self.faceList)

                        # 矩形框
                        for kk, d in enumerate(faces):
                            # 绘制矩形框
                            cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]),
                                          (0, 255, 255), 2)

                    # 在人脸框下面写人脸名字
                    for i in range(len(faces)):
                        cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

            # print("Name list now:", name_namelist, "\n")

            cv2.putText(img_rd, "Press 'q': Quit", (20, 450), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
            cv2.putText(img_rd, "Face Recognition", (20, 40), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

            # 窗口显示
            cv2.imshow("camera", img_rd)

        # 释放摄像头
        cap.release()

        # 删除建立的窗口
        cv2.destroyAllWindows()

        #self.conn.close()

    # 计算两个128D向量间的欧式距离
    def return_euclidean_distance(self,feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        print("e_distance: ", dist)

        if dist > 0.4:
            return "diff"
        else:
            return "same"



    def get_person(self,path):
        # 打开execl
        workbook = xlrd.open_workbook(path)
        # 根据sheet索引或者名称获取sheet内容
        Data_sheet = workbook.sheets()[0]  # 通过索引获取
        # Data_sheet = workbook.sheet_by_index(0)  # 通过索引获取
        # Data_sheet = workbook.sheet_by_name(u'名称')  # 通过名称获取
        rowNum = Data_sheet.nrows  # sheet行数
        colNum = Data_sheet.ncols  # sheet列数

        # 获取所有单元格的内容
        for i in range(1,rowNum):
            rowlist = []
            for j in range(colNum):
                rowlist.append(Data_sheet.cell_value(i, j))
            self.personList.append(rowlist)
        return  self.personList

        # 返回一张图像多张人脸的 128D 特征
        def get_128d_features(self,img_gray):
            self.faces = self.detector(img_gray, 1)
            if len(self.faces) != 0:
                face_des = []
                for i in range(len(self.faces)):
                    shape = self.predictor(img_gray, self.faces[i])
                    face_des.append(self.facerec.compute_face_descriptor(img_gray, shape))
            else:
                face_des = []
            return face_des


