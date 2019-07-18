import pickle

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# t = numpy.linspace(0,8063, num=numpy.floor(8063 + 1))
# print(t)
# print(t.shape)
# a = signal.chirp(t = t, f0 = 0, t1 = 8063, f1 = 200, method='linear', phi=0)
# print(a.shape)
# # plt.plot(a)
# # plt.show()
# f, t, Zxx = signal.stft(a, fs = 64, nperseg=128)
# plt.pcolormesh(t, f, numpy.abs(Zxx), vmin=0)
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

#
# t = numpy.linspace(0,10, num=numpy.floor(10*10000 + 1))
#
# a = signal.chirp(t = t, f0 = 1, t1 = 10, f1 = 100, method='linear', phi=0)
# plt.plot(a)
# plt.show()
#
# f, t, Zxx = signal.stft(a, fs = 1000, nperseg=128)
# plt.pcolormesh(t, f, numpy.abs(Zxx), vmin=0)
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

def plot(xx):
    # single frequency signal
    N = 500
    xf = np.fft.fft(xx)
    xf_abs = np.fft.fftshift(abs(xf))
    axis_xf = np.linspace(-N/2,N/2-1,num=N)
    plt.title(u'频率为5Hz的正弦频谱图',fontproperties='SimHei')
    plt.plot(xf_abs)
    plt.axis('tight')
    plt.show()

t = np.linspace(0,63, num=np.floor(63*128))
# 1000为采样率
# print(t.shape)
a = signal.chirp(t = t, f0 = 40, t1 = 63, f1 = 150, method='linear', phi=np.random.normal(0,100, size=(1,)))

fname = "./data_preprocessed_python\s01.dat"     #C:/Users/lumsys/AnacondaProjects/Emo/
f = open(fname, 'rb')
x = pickle.load(f, encoding='latin1')
for tr in range(40):
    a = x['data'][tr][39][:]
    plot(a)




# f, t, Zxx = signal.stft(a, fs = 128, nperseg=128)
# plt.pcolormesh(t, f, np.abs(Zxx))
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
