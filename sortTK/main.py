# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import tkinter.messagebox
from tkinter import *
from PIL import Image,ImageTk
from tkinter import ttk
from ttkthemes import themed_tk as tk
import cv2
import os
import gc
from multiprocessing import Process, Manager
import draw
from face_recognition import get_faces_from_camera as gf
from face_recognition import get_features_into_CSV as csv
from face_recognition import face_reco_from_camera as fr
from tkinter.filedialog import askdirectory
from deep_sort_yolo import use_camera_fast

import tkinter.font as tkFont
def fareco(ip):
    fr.Reco(ip)
def write(stack, cam, top) :
    """
    :param cam: 摄像头参数
    :param stack: Manager.list对象
    :param top: 缓冲栈容量
    :return: None
    """
    print('Process to write: %s' % os.getpid())
    cap = cv2.VideoCapture(cam)
    state = 0
    while True:

        _, img = cap.read()
        if _:
            stack.append(img)
            if state == 0:
                I = Image.fromarray(img[..., ::-1])
                I.save('frame.jpg')
                state = 1
            # 每到一定容量清空一次缓冲栈
            # 利用gc库，手动清理内存垃圾，防止内存溢出
            if len(stack) >= top:
                del stack[:]
                gc.collect()
def read(stack) :
    print('Process to read: %s' % os.getpid())

    while True:
        if len(stack) != 0:
            frame = stack.pop()

            cv2.imshow("img", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

class basedesk():
    def __init__(self, master):
        self.root = master
        self.root.config()
        self.root.title("变电站作业管控")
        self.root.get_themes()
        self.root.set_theme("radiance")
        interface_first(self.root)

class basedesk2():
    def __init__(self, master):
        self.root = master
        self.root.config()
        self.root.title("变电站作业管控")
        self.root.get_themes()
        self.root.set_theme("radiance")
        initface_sort(self.root)
class interface_first():
    def __init__(self,master):
        self.master = master
        self.interface_first = Frame(self.master)
        self.interface_first.pack()
        self.topframe = Frame(self.interface_first)
        self.topframe.pack()
        self.middleframe = Frame(self.interface_first)
        self.middleframe.pack()
        self.bottomframe = Frame(self.interface_first)
        self.bottomframe.pack()
        self.label1 = tkinter.Label(self.topframe, text="人脸识别摄像头ip", fg="black", font=("MS Sans Serif", 12), width=18,height=2, )
        self.label1.pack(side=LEFT)
        self.ip1 = Entry(self.topframe, bd=12)
        self.ip1.pack(side=RIGHT)
        self.label2 = tkinter.Label(self.middleframe, text="作业管控摄像头ip", fg="black", font=("MS Sans Serif", 12), width=18,height=2, )
        self.label2.pack(side=LEFT)
        self.ip2 = Entry(self.middleframe, bd=12)
        self.ip2.pack(side=RIGHT)
        self.okBtn = Button(self.bottomframe, text='确定', font=("MS Sans Serif", 12),command=self.change_reco)
        self.okBtn.pack()

    def change_reco(self):
        global camera1
        camera1 = str(self.ip1.get())
        global camera2
        camera2 = str(self.ip2.get())
        cap = cv2.VideoCapture(camera1)
        while True:
            _, img = cap.read()
            if _:
                draw.draw(img)
                picture = Image.fromarray(img[..., ::-1])

                picture.save('frame1.jpg')
            break
        self.interface_first.destroy()
        initface_reco(self.master)




class initface_reco():

    def __init__(self, master):

        self.ft = tkFont.Font(family='MS Sans Serif', size=13, weight=tkFont.NORMAL)

        self.master = master
        # 基准界面initface
        self.initface_reco = Frame(self.master)
        self.initface_reco.pack()
        self.statusbar = ttk.Label(self.initface_reco, text="Welcome", relief=SUNKEN, anchor=W,
                                   font='Times 10 italic')
        self.statusbar.pack(side=BOTTOM, fill=X)
        # Create the menubar
        self.menubar = Menu(self.initface_reco)
        self.master.config(menu=self.menubar)
        # Create the submenu
        self.subMenu = Menu(self.menubar, tearoff=0)
        playlist = []
        self.menubar.add_cascade(label="选择功能", menu=self.subMenu,font=self.ft)
        self.subMenu.add_command(label="人脸识别", font=self.ft,command='')
        self.subMenu.add_command(label="作业管控", font=self.ft,command='')
        self.subMenu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="帮助", font=self.ft,menu=self.subMenu)
        self.subMenu.add_command(label="使用说明", font=self.ft,command='')
        self.subMenu.add_command(label="开发者信息", font=self.ft,command='')
        # 顶部
        self.topframe = Frame(self.initface_reco)
        self.topframe.pack(side=TOP)
        #self.label1 = tkinter.Label(self.topframe, text="人脸识别摄像头ip", fg="black", font=("MS Sans Serif", 12), width=18,height=2, )
        self.mainlabel = tkinter.Label(self.topframe, text="人脸识别打卡", fg="black", font=self.ft, width=30,
                                       height=2, )
        self.mainlabel.pack(side=TOP)
        # 左边
        self.leftframe = Frame(self.initface_reco)
        self.leftframe.pack(side=LEFT, padx=30, pady=5)
        self.linelabe = tkinter.Label(self.leftframe, text="画面预览", fg="black", font=self.ft, width=15, height=2, )
        self.linelabe.pack(side=TOP)
        self.load = Image.open('frame1.jpg').resize((324, 216))
        self.render = ImageTk.PhotoImage(self.load)
        self.img = Label(self.leftframe, image=self.render)
        self.img.pack()


        # 右边
        self.rightframe = Frame(self.initface_reco)
        self.rightframe.pack(pady=80)

        self.r_topframe = Frame(self.rightframe)
        self.r_topframe.pack()

        self.tasklabel = ttk.Label(self.r_topframe, text='工作任务 : --:--',font=self.ft)
        self.tasklabel.pack(pady=5)


        self.uploadBtn = Button(self.r_topframe, text='上传',font=self.ft, command=self.upload)
        self.uploadBtn.pack()
        muted = FALSE
        self.r_middleframe = Frame(self.rightframe)
        self.r_middleframe.pack(pady=30, padx=30)
        text1 ="人脸识别摄像头：" +'\n'+ camera1
        self.facelabel = ttk.Label(self.r_middleframe, text=text1,font=self.ft)
        self.facelabel.grid(row=0, column=0, padx=10)
        text2 = "作业管控摄像头："  +'\n'+ camera2
        self.sortlabel = ttk.Label(self.r_middleframe, text=text2,font=self.ft)
        self.sortlabel.grid(row=1, column=0, padx=10)

        # Bottom Frame for volume, rewind, mute etc.
        self.r_bottomframe = Frame(self.rightframe)
        self.r_bottomframe.pack()
        self.getfaceBtn = Button(self.r_bottomframe, text='录入人脸', font=self.ft,command=self.get_face)
        self.getfaceBtn.grid(row=0, column=0)
        self.intoCSV = Button(self.r_bottomframe, text='写入特征', font=self.ft, command=self.intoCSV)
        self.intoCSV.grid(row=0, column=1)

        self.nextBtn = Button(self.r_bottomframe, text='进入作业管控', font=self.ft,command=self.change_sort)
        self.nextBtn.grid(row=0, column=2)
    def intoCSV(self):
        csv.IntoCSV()
    def upload(self):
        d = os.path.dirname(__file__)
        file_path = tkinter.filedialog.askopenfilename(title=u'选择文件', initialdir=(os.path.expanduser(d)))
        import shutil
        file_path = file_path
        new = os.path.dirname(d)+'/face_recognition'
        shutil.copy(file_path, new)
        tkinter.messagebox.showinfo('提示', '上传成功')
        self.tasklabel["text"] = '工作任务 : '+ file_path.split('/')[-1]
    def get_face(self):
        gf.GetFace(camera1)
    def change_sort(self):
        #rtsp://admin:WFGMYS@192.168.1.102:554//Streaming/Channels/1

        cap = cv2.VideoCapture(camera2)
        while True:
            _, img = cap.read()
            if _:
                draw.draw(img)
                picture = Image.fromarray(img[..., ::-1])

                picture.save('frame2.jpg')
            break
        self.initface_reco.destroy()
        initface_sort(self.master)



class initface_sort():
    def __init__(self, master):
        self.master = master
        # 基准界面initface
        self.initface_sort = Frame(self.master)
        self.initface_sort.pack()
        self.statusbar = ttk.Label(self.initface_sort, text="Welcome", relief=SUNKEN, anchor=W, font='Times 10 italic')
        self.statusbar.pack(side=BOTTOM, fill=X)
        # Create the menubar
        self.menubar = Menu(self.initface_sort)
        self.master.config(menu=self.menubar)
        # Create the submenu
        self.subMenu = Menu(self.menubar, tearoff=0)
        playlist = []
        self.ft = tkFont.Font(family='MS Sans Serif', size=13, weight=tkFont.NORMAL)
        self.menubar.add_cascade(label="选择功能", menu=self.subMenu,font = self.ft)
        self.subMenu.add_command(label="人脸识别", font = self.ft,command='')
        self.subMenu.add_command(label="作业管控", font = self.ft,command='')
        self.subMenu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="帮助", font = self.ft,menu=self.subMenu)
        self.subMenu.add_command(label="使用说明", font = self.ft,command='')
        self.subMenu.add_command(label="开发者信息", font = self.ft,command=self.about_us)
        #顶部
        self.topframe = Frame(self.initface_sort)
        self.topframe.pack(side=TOP)
        self.mainlabel = tkinter.Label(self.topframe, text="变电站移动目标跟踪及作业管控", fg="black", font = self.ft, width=30, height=2, )
        self.mainlabel.pack(side=TOP)
        #左边
        self.leftframe = Frame(self.initface_sort)
        self.leftframe.pack(side=LEFT, padx=30, pady=5)
        self.linelabe = tkinter.Label(self.leftframe, text="警戒线", fg="black", font = self.ft, width=15, height=2, )
        self.linelabe.pack(side=TOP)
        self.load = Image.open('frame2.jpg').resize((324, 216))
        self.render = ImageTk.PhotoImage(self.load)
        self.img = Label(self.leftframe, image=self.render)
        self.img.pack()
        self.addBtn = Button(self.leftframe, text="+ 更新", font = self.ft,command='')
        self.addBtn.pack(side=LEFT)
        self.delBtn = Button(self.leftframe, text="- 删除", font = self.ft,command='')
        self.delBtn.pack(side=RIGHT)
        #右边
        self.rightframe = Frame(self.initface_sort)
        self.rightframe.pack(pady=80)

        self.r_topframe = Frame(self.rightframe)
        self.r_topframe.pack()
        self.patten = 0
        self.lengthlabel = ttk.Label(self.r_topframe, font = self.ft,text='已选模式 : --:--')
        self.lengthlabel.pack(pady=5)

        self.currenttimelabel = ttk.Label(self.r_topframe, font = self.ft,text='模式功能 : --:--', relief=GROOVE)
        self.currenttimelabel.pack()
        muted = FALSE
        self.r_middleframe = Frame(self.rightframe)
        self.r_middleframe.pack(pady=30, padx=30)
        self.Btn1 = Button(self.r_middleframe, text="模式一", font = self.ft,command=self.btn1)
        self.Btn1.grid(row=0, column=0, padx=10)

        self.Btn2 = Button(self.r_middleframe, text="模式二", font = self.ft,command=self.btn2)
        self.Btn2.grid(row=0, column=1, padx=10)

        self.Btn3 = Button(self.r_middleframe, text="模式三", font = self.ft,command=self.btn3)
        self.Btn3.grid(row=0, column=2, padx=10)

        # Bottom Frame for volume, rewind, mute etc.
        self.r_bottomframe = Frame(self.rightframe)
        self.r_bottomframe.pack()
        self.startBtn = Button(self.r_bottomframe, text='开始', font = self.ft,command=self.sort_frame)
        self.startBtn.grid(row=0, column=0)

        self.stopBtn = Button(self.r_bottomframe, text='结束', font = self.ft,command='')
        self.stopBtn.grid(row=0, column=1)


    def about_us(self,):
        tkinter.messagebox.showinfo('开发者信息', 'lyk19940625@qq.com')

    def btn1(self,):
        self.patten = 1
        self.lengthlabel["text"] = '模式一'
        self.currenttimelabel['text'] = '移动目标跟踪 越界检测'

    def btn2(self,):
        self.patten = 2
        self.lengthlabel["text"] = '模式二'
        self.currenttimelabel['text'] = '移动目标跟踪 越界检测  异常穿戴检测'

    def btn3(self,):
        self.patten = 3
        self.lengthlabel["text"] = '模式三'
        self.currenttimelabel['text'] = '移动目标跟踪 越界检测  异常穿戴检测 姿态检测'

    def sort_frame(self):
        if self.patten == 0:
            tkinter.messagebox.showinfo('提示', '请选择模式')
        elif self.patten == 1:
            self.sort_frame1()
        elif self.patten == 2:
            self.sort_frame2()
        elif self.patten == 3:
            self.sort_frame3()

    def sort_frame(self):
        '''

        可以加入人脸识别进程，或发送启动信号
        facereco = Process(target=fareco, args=(camera1))
        facereco.start()
        '''
        q = Manager().list()
        pw = Process(target=use_camera_fast.write, args=(q, camera2, 100))
        pr = Process(target=use_camera_fast.read, args=(q,))

        # 启动子进程pw，写入:
        pw.start()
        # 启动子进程pr，读取:
        pr.start()

        # 等待pr结束:
        pr.join()

        # pw进程里是死循环，无法等待其结束，只能强行终止:
        pw.terminate()
    def sort_frame2(self):
        '''
        写模式三进程

        '''
        print()
    def sort_frame3(self):
        '''写模式三进程'''


        print()



if __name__ == '__main__':

    root = tk.ThemedTk()
    basedesk(root)
    root.mainloop()
