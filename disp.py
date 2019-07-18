import json

import pickle
import threading
import time
# file_x = './data/features_noise.dat'
# file_y = './data/label_class_0.dat'
#
# X = numpy.genfromtxt(file_x, delimiter=' ')
# y = numpy.genfromtxt(file_y, delimiter=' ')
# Y = numpy.array([i for i in range(X.shape[1])])
# print(X.shape[1])
# print(X.shape[0])
# print(Y)



# from itertools import combinations
#
# import numpy
#
# X = numpy.array([[1,2,3,4,5],[2,3,4,5,3]])
# Y = numpy.array([i for i in range(X.shape[1])])
# print(Y)
# test = combinations(Y[0:3],r =3)
#
# for el in test:
#     print(el)

#!/usr/bin/python3

import _thread
import time
#
# # 为线程定义一个函数
# def print_time( threadName, delay):
#    count = 0
#    while count < 5:
#       time.sleep(delay)
#       count += 1
#       print ("%s: %s" % ( threadName, time.ctime(time.time()) ))
#
# def f( str ="hello"):
#
#     count = 0
#     while count < 5:
#         time.sleep(2)
#         print(str)
#
# # 创建两个线程
# try:
#    _thread.start_new_thread(f,("Thread-1",))
#    _thread.start_new_thread(f,("Thread-2",))
# except:
#    print ("Error: 无法启动线程")
#
# while 1:
#    pass

# exitFlag = 0
#
# class myThread (threading.Thread):
#     def __init__(self, threadID, name, counter):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.name = name
#         self.counter = counter
#     def run(self):
#         print ("开始线程：" + self.name)
#         print_time(self.name, self.counter, 5)
#         print ("退出线程：" + self.name)
#
# def print_time(threadName, delay, counter):
#     while counter:
#         if exitFlag:
#             threadName.exit()
#         time.sleep(delay)
#         print ("%s: %s" % (threadName, time.ctime(time.time())))
#         counter -= 1
#
# # 创建新线程
# thread1 = myThread(1, "Thread-1", 1)
# thread2 = myThread(2, "Thread-2", 2)
#
# # 开启新线程
# thread1.start()
# thread2.start()
# thread1.join()
# thread2.join()
# print ("退出主线程")


# <class 'dict'>
# {'2': ((0, 1), 0.55625), '3': ((0, 1, 2), 0.56796875)}

f1 = open('./data/SBS/clear_valence_18fea.json','r')
f2 = open('./data/SBS/noise_valence.json','r')
f3 = open('./data/SBS/clear_arousal_18fea.json','r')
f4 = open('./data/SBS/noise_arousal.json','r')
f1c = json.load(f1)
f2c = json.load(f2)
f3c = json.load(f3)
f4c = json.load(f4)
for i in range(2,18):
    print(str(i)+" features:   ")
    print("valence:")
    print("clear")
    print(f1c[str(i)])
    # print("noise")
    # print(f2c[str(i)])
    print("arousal:")
    print("clear")
    print(f3c[str(i)])
    # print("noise")
    # print(f4c[str(i)])
    # print()
f1.close()
f2.close()
f3.close()
f4.close()
