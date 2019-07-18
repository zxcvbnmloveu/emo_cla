import csv
import math
import numpy
import pickle

import numpy as np
from scipy import stats, signal
from matplotlib import pyplot as plt

end = 8064
x = np.arange(0,end)
y =  2  * x +  5
#
nLabel, nTrial, nUser, nChannel, nTime  = 4, 40, 32, 40, 8064
print("Program started"+"\n")

fname = "./data_preprocessed_python\s06.dat"     #C:/Users/lumsys/AnacondaProjects/Emo/
f = open(fname, 'rb')
data = pickle.load(f, encoding='latin1')
print(fname)
data1 = data['data'][0][33][0:end]
data2 = data['data'][0][35][0:end]
data3 = data['data'][0][36][0:end]
data4 = data['data'][0][37][0:end]
# plt.title("Matplotlib demo")
plt.xlabel("x axis caption")
plt.ylabel("y axis caption")
plt.subplot(1,4,1)
plt.title("36")
plt.plot(x,data1)
plt.subplot(1,4,2)
plt.title("37")
plt.plot(x,data2)
plt.subplot(1,4,3)
plt.title("38")
plt.plot(x,data3)
plt.subplot(1,4,4)
plt.title("39")
plt.plot(x,data4)
plt.show()

t = np.linspace(0,63, num=np.floor(63*128))

# min = 1000
# max = 2000
# rand = min + (max-min)*np.random.random()
# noise = np.random.normal(0, rand, size=(8064,))
# plt.plot(x,noise)
# plt.show()
#
# data3_noise = data3 + noise
#
# plt.title("Matplotlib demo")
# plt.xlabel("x axis caption")
# plt.ylabel("y axis caption")
# plt.subplot(1,2,1)
# plt.title("clean")
# plt.plot(x[0:1024],data3[0:1024])
# plt.subplot(1,2,2)
# plt.title("noise")
# plt.plot(x[0:1024],data3_noise[0:1024])
# plt.show()

# oringin = []
# with open('./data_original\\new_edition\\features\\0.csv') as csvfile:
#     csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
#     for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
#         oringin.append(row)
#     oringin = numpy.array(oringin)
#     print(oringin.shape)
# oringin = oringin.reshape((-1))
#
# preposses = []
# with open('./data\csv_features\\features\\0.csv') as csvfile:
#     csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
#     for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
#         preposses.append(row)
#     preposses = numpy.array(preposses)
#     print(preposses.shape)
# preposses = preposses.reshape((-1))
#
# plt.title("Matplotlib demo")
# plt.xlabel("x axis caption")
# plt.ylabel("y axis caption")
# plt.subplot(1,2,1)
# plt.title("clean")
# plt.plot(x[0:7697],oringin)
# plt.subplot(1,2,2)
# plt.title("noise")
# plt.plot(x,preposses)
# plt.show()
