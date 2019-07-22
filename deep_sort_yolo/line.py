import math
import operator
import csv
def cross(p1,p2,p3):#跨立实验
    x1=p2[0]-p1[0]
    y1=p2[1]-p1[1]
    x2=p3[0]-p1[0]
    y2=p3[1]-p1[1]
    return x1*y2-x2*y1

def getLine2(track,line1):
    track_dic = {}
    line2s = []
    for t in range(len(track)):
        dis = int(math.sqrt(pow(track[t][0] - line1[0][0], 2) + pow(track[t][1] - line1[0][1], 2)))
        track_dic[dis] = t
    index = sorted(track_dic.items(), key=operator.itemgetter(0), reverse=False)[0][1]
    p1 = track[index]

    if index == 0:
        p2 = p1
        p3 = track[index + 1]

    elif index == len(track)-1:
        p2 = track[index - 1]
        p3 = p1
    else:
        p2 = track[index-1]
        p3 = track[index + 1]
    line2_1 = [p2,p1]
    line2_2 = [p1,p3]
    line2s.append(line2_1)
    line2s.append(line2_2)
    return line2s


def IsIntersec(line1,line2): #判断两线段是否相交
    p1 = line2[0] # 取四点坐标
    p2 = line2[1]
    p3 = line1[0]
    p4 = line1[1]


    #快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if(max(p1[0],p2[0])>=min(p3[0],p4[0])    #矩形1最右端大于矩形2最左端
    and max(p3[0],p4[0])>=min(p1[0],p2[0])   #矩形2最右端大于矩形最左端
    and max(p1[1],p2[1])>=min(p3[1],p4[1])   #矩形1最高端大于矩形最低端
    and max(p3[1],p4[1])>=min(p1[1],p2[1])): #矩形2最高端大于矩形最低端

    #若通过快速排斥则进行跨立实验
        if(cross(p1,p2,p3)*cross(p1,p2,p4)<=0
           and cross(p3,p4,p1)*cross(p3,p4,p2)<=0):
            D='有交点'
        else:
            D='无交点'
    else:
        D='无交点'
    return D

def IsIntersec2(track,line1):
    sate = '无交点'
    lines = getLine2(track,line1)
    for line2 in lines:
        if IsIntersec(line1,line2) == '有交点':
            sate = IsIntersec(line1,line2)
    return sate

def writeline(line):
    with open('line.csv','w') as file:
        f_csv = csv.writer(file)
        f_csv.writerows(line)


def readline():
    lines = []

    with open('line.csv','r') as file:
        csv_data = csv.reader(file)
        for row in csv_data:
            line = []
            for r in row:
                line.append(int(r))
            lines.append(line)
    return lines

#写入警戒线
# line = [(1095,632),(135,630),(263,411),(1123,419),(1276,530)]
# writeline(line)