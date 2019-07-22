class Worker:


    def __init__(self):
        self.id = ''
        self.wear = {'no helmet':[],'no work cloths':[],'unsafe wear':[]}
        self.in_time = ''
        self.out_time = ''
        self.track_point = []
        self.transboundary = []
        self.current_point = (0, 0)
        self.current_footL = (0, 0)
        self.current_footR = (0, 0)
        self.next_point = (0, 0)
        self.previous_point = (0, 0)
    def set(self,id,in_time,current_point):
        self.id = id
        self.in_time = in_time
        self.current_point = current_point
        self.track_point.append(current_point)

"""
def update_worker_ct(key,point):
    workers[key].current_point = point
    workers[key].track_point.append(point)

import time
import math

#思路
#获得刷脸进入顺序
face_info = ['1','2','3','4']
count = 0
worker_dict = {}
#新建识别人员列表
workers = {}
#获得识别框信息
box_title = '类别'
box_point = ''
#如果类别person,判断是否在workers里，如果没有则加入，如有就更新；如果不是person，更新最近的worker，type设置类别。
if box_title[:5] == 'person':

    if len(workers) == 0:
        localtime = time.asctime(time.localtime(time.time()))
        worker_dict[box_title] = face_info[count]
        count = count + 1
        workers[box_title] = worker(worker_dict[box_title],localtime,box_point)
    else:
        update_worker_ct(box_title,box_point)

        if len(workers[box_title].track_point) >=20:
            workers[box_title].previous_point = workers[box_title].track_point[-5]

else:
    min, temp = 999, ''
    for w in workers:
        dis = math.sqrt(
            ((workers[w].current_point[0] - box_point[0]) ** 2) + ((workers[w].current_point[1] - box_point[1]) ** 2))
        if min > dis:
            min = dis
            temp = w
    #更新
    update_worker_ct(temp, box_point)
    workers[temp].type = box_title
    if len(workers[temp].track_point) >= 20:
        workers[temp].previous_point = workers[box_title].track_point[-5]



"""





"""
后续任务：
判断目标个数
加入卡尔曼滤波器
    如果识别替换next，这帧没有append，目标再出现，把预测加入track
加入姿态点更新
考虑多视角
"""
