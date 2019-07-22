#综述
使用该作业现场安全生产智能管控平台来实现变电站的安全生产的智能化管理，通过人脸识别功能进行人员的考勤；
通过人员、车辆的检测和识别来实现变电站的智能化管理；通过安全行为识别和安全区域报警功能来实现对变电站内人员和设备安全的监督；
![avatar](https://github.com/lyk19940625/WorkControl/blob/master/output.png)
# 移动目标跟踪

## 介绍
项目利用DeepSort算法实现作业现场移动目标跟踪定位。
论文参考：SIMPLE ONLINE AND REALTIME TRACKING WITH A DEEP ASSOCIATION METRIC

代码参考：https://github.com/nwojke/deep_sort

DeepSort是在Sort目标追踪基础上的改进。引入了在行人重识别数据集上离线训练的深度学习模型，
在实时目标追踪过程中，提取目标的表观特征进行最近邻匹配，可以改善有遮挡情况下的目标追踪效果。
同时，也减少了目标ID跳变的问题。

算法关键点为：
1、在计算detections和tracks之间的匹配程度时，使用了融合的度量方式。
包括卡尔曼滤波中预测位置和观测位置在马氏空间中的距离 和 bounding boxes之间表观特征的余弦距离。
2、其中bounding box的表观特征是通过一个深度网络得到的128维的特征
3、在匈牙利匹配detections和tracks时，使用的是级联匹配的方式。这里要注意的是，
并不是说级联匹配的方式就比global assignment效果好，而是因为本文使用kalman滤波计算运动相似度的缺陷导致使用级联匹配方式效果更好。

## 依赖
* NumPy
* sklearn
* OpenCV

## 描述

在包中deep_sort是主要的跟踪代码：

* detection.py：检测基类。
* kalman_filter.py：卡尔曼滤波器实现和图像空间滤波的具体参数化。
* linear_assignment.py：此模块包含最低成本匹配和匹配级联的代码。
* iou_matching.py：此模块包含IOU匹配指标。
* nn_matching.py：最近邻匹配度量的模块。
* track.py：轨道类包含单目标轨道数据，例如卡尔曼状态，命中数，未命中，命中条纹，相关特征向量等。
* tracker.py：这是多目标跟踪器类。

# 人脸识别
## 介绍
借助 Dlib 库捕获摄像头中的人脸，提取人脸特征，通过计算特征值之间的欧氏距离来和预存的人脸特征进行对比，
判断是否匹配，达到人脸识别的目的。

## 依赖
* Dlib
* OpenCV

## 描述

* get_face_from_camera.py / 脸注册录入，将检测到的人脸图像存入相应的文件夹，形成特征图像数据集。

* get_features_into_CSV.py / 提取出 128D 特征，然后计算出某人人脸数据的特征均值存入 CSV。

* face_reco_from_camera.py / 调用摄像头，捕获摄像头中的人脸，如果检测到人脸，将摄像头中的人脸提取出 128D 的特征，
然后和之前录入人脸的 128D 特征 进行计算欧式距离，判断身份。

# 安全措施异常检测
## 介绍
使用yolo v3框架训练自己的数据集完成安全措施的检测。分类包括未戴安全帽、未穿工作服、无安全措施、符合安全规范四种类型。
最终识别率在百分之75左右。

参考博客：https://blog.csdn.net/Patrick_Lxc/article/details/80615433

## 依赖
* Python 3.5.2
* Keras 2.1.5
* tensorflow 1.6.0

## 描述
yolo关键点简述

1、端到端，输入图像，一次性输出每个栅格预测的一种或多种物体

2、坐标x,y代表了预测的bounding box的中心与栅格边界的相对值。
 坐标w,h代表了预测的bounding box的width、height相对于整幅图像（或者栅格）width,height的比例。 

3、每个格子可以预测B个bounding box，但是最终只选择只选择IOU最高的bounding box作为物体检测输出，

# 二维人体姿态估计
## 介绍
项目采用OpenPose算法对作业人员的姿态进行识别检测；OpenPose人体姿态识别可以实现人体动作、面部表情、手指运动等姿态估计。适用于单人和多人，具有极好的鲁棒性。是世界上首个基于深度学习的实时多人二维姿态估计应用。
其核心是利用Part Affinity Fields（PAFs）的自下而上的人体姿态估计算法。自下而上算法是先得到关键点位置再获得骨架；流程是首先通过预测人体关键点的热点图获得人体关键点的位置，在热点图中每个人体关键点上都有一个高斯的峰值，代表神经网络确定这里有一个人体的关键点。在得到检测结果之后，对关键点检测结果进行连接，推测出每个关键点具体是属于哪个人。


## 描述
使用TensorFlow轻松实现openpose。

只使用基本的python，所以代码很容易理解。


Original Repo（Caffe）：https：//github.com/CMU-Perceptual-Computing-Lab/openpose。

Dataloader和Post-processing代码来自tf-pose-estimation。

![avatar](https://github.com/lyk19940625/WorkControl/blob/master/unofficial-openpose/graph_run.png)