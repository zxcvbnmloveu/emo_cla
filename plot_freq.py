import numpy as np  # 导入一个数据处理模块
import pylab as pl  # 导入一个绘图模块，matplotlib下的模块
import matplotlib as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import pickle

from scipy.signal import chirp
from sklearn.preprocessing import StandardScaler


def noise_gen(f0, len):
    t = np.arange(0, len, 1.0 / 128)
    w = chirp(t, f0, f1=0.4, t1=10, method="linear")
    # pl.plot(t, w)
    # pl.show()
    return w


# ‘linear’, ‘quadratic’, ‘logarithmic’, ‘hyperbolic’

def p_t_f(t_len, sampling_rate, Signal, ):
    N = (t_len - 0) * sampling_rate
    axis_x = np.arange(0, t_len, 1.0 / sampling_rate)
    pl.figure()
    pl.subplot(211)
    pl.plot(axis_x, Signal[:N])
    pl.title(u'time', fontproperties='SimHei')
    pl.axis('tight')

    xf = np.fft.fft(Signal)[:N//2+1]
    # xfp = 20*np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    axis_xf = np.linspace(0, sampling_rate/2, num=N//2+1)
    pl.subplot(212)
    pl.title(u'freq', fontproperties='SimHei')
    my_y_ticks = np.arange(0, 1000000, 100000)
    # pl.yticks(my_y_ticks)
    # print(axis_xf.shape)
    pl.plot(axis_xf[:1000], abs(xf)[:1000])
    pl.axis('tight')
    # pl.show()



if __name__ == '__main__':


    # s1 = noise_gen(f0=10, len=63)
    # p_t_f(sampling_rate=128, Signal=s1, t_len=10)
    # s2 = noise_gen(f0 = 0.5, len = 21)
    # s3 = noise_gen(f0 = 0.5, len = 21)
    # noise36 = np.append(s1, s2)
    # noise36 = np.append(noise36, s3)
    #
    # s1 = noise_gen(f0 = 0.2, len = 21)
    # s2 = noise_gen(f0 = 0.2, len = 21)
    # s3 = noise_gen(f0 = 0.2, len = 21)
    # noise38 = np.append(s1, s2)
    # noise38 = np.append(noise38, s3)
    # # plot_wf(sampling_rate = 128, Signal= noise, fft_size = 8064, t_len = 63, name = "data32")
    fname = "./data_preprocessed_python\s06.dat"  # C:/Users/lumsys/AnacondaProjects/Emo/
    f = open(fname, 'rb')
    data = pickle.load(f, encoding='latin1')
    # data32 = data['data'][0][32]
    # data33 = data['data'][0][33]
    # data36 = data['data'][0][36]


    data_s = data['data'][0][33][0+2024:1024+2024]
    print(data_s.shape)
    data_s = data_s.reshape(-1, 1)
    data_s = StandardScaler().fit_transform(data_s).reshape(1024)
    print(data_s.shape)

    data33 = data_s

    data_s = data['data'][0][34][0:1024]
    print(data_s.shape)
    data_s = data_s.reshape(-1, 1)
    data_s = StandardScaler().fit_transform(data_s).reshape(1024)
    print(data_s.shape)
    data34 = data_s
    gauss = np.random.normal(0, 1, size=(256,))
    sos = signal.butter(4, 5, 'high',fs=128, output='sos')
    gauss = signal.sosfilt(sos, gauss)
    # sos = signal.butter(4, 10, 'low',fs=128, output='sos')
    # gauss = signal.sosfilt(sos, gauss)

    noise = np.append(0.96*gauss, 0.8*gauss)
    noise = np.append(noise, 0.78*gauss)
    noise = np.append(noise, 0.6*gauss)
    print(noise.shape)
    #
    a = np.linspace(-1, 2, num=10)
    b = np.linspace(2, 4, num=15)
    c = np.linspace(2.1, -1.1, num=25)
    d = np.linspace(1, -0.8, num=30)
    feng = np.append(a,b)
    feng = np.append(feng,c)
    feng = np.append(feng,d)
    a = np.zeros(635)
    b = np.zeros(309)
    feng = np.append(a,feng)
    feng = np.append(feng,b)

    p_t_f(sampling_rate=128, Signal=data33[:1024], t_len=8)
    p_t_f(sampling_rate=128, Signal=data33[:1024] + 0.35*noise, t_len=8)
    # p_t_f(sampling_rate=128, Signal=data33[:1024] + 0.3*noise + 0.5 * feng, t_len=8)

    pl.show()

    # adr = "C:/Users/cuzzw/Desktop/pic3/GSR.txt"
    # fout_data = open(adr,'w')
    # for i in data33[:1024]:
    #     fout_data.write(str(i) + " ")
    # fout_data.close()
    #
    # adr = "C:/Users/cuzzw/Desktop/pic3/GSR1.txt"
    # fout_data = open(adr,'w')
    # for i in data33[:1024] + 0.3*noise:
    #     fout_data.write(str(i) + " ")
    # fout_data.close()
    #
    adr = "C:/Users/cuzzw/Desktop/pic3/33.txt"
    fout_data = open(adr,'w')
    for i in data33[:1024]:
        fout_data.write(str(i) + " ")
    fout_data.close()

    # p_t_f(sampling_rate=128, Signal=data_s + 40 * MA, t_len=8)
    # pl.show()
    # noise = data['data'][0][34]
    # plot_wf(sampling_rate = 128, Signal= data32, fft_size = 1024, t_len = 8, name = "data32")
    # plot_wf(sampling_rate = 128, Signal= data33, fft_size = 1024, t_len = 8, name = "data33" )
    # plot_wf(sampling_rate = 128, Signal= data36, fft_size = 1024, t_len = 8, name = "data36" )
    # plot_wf(sampling_rate = 128, Signal= data38, fft_size = 1024, t_len = 8, name = "data38" )
    # plot_wf(sampling_rate = 128, Signal= data32 + 2.5 * data['data'][0][34], fft_size = 1024, t_len = 8, name = "data32w" )
    # plot_wf(sampling_rate = 128, Signal= data33 + 2.5 * data['data'][0][34], fft_size = 1024, t_len = 8, name = "data33w" )
    # plot_wf(sampling_rate = 128, Signal= data36 + 2000 * s1 + np.random.normal(0,300, size=(8064,)), fft_size = 1024, t_len = 8, name = "data36w" )
    # plot_wf(sampling_rate = 128, Signal= data38 + 2000 * noise38 + np.random.normal(0,100, size=(8064,)), fft_size = 1024, t_len = 8, name = "data38w" )
    # plot_wf(sampling_rate = 128, Signal= data32 + 15 * data['data'][0][34], fft_size = 1024, t_len = 8, name = "data32s" )
    # plot_wf(sampling_rate = 128, Signal= data33 + 15 * data['data'][0][34], fft_size = 1024, t_len = 8, name = "data33s" )
    # s1s = noise_gen(f0 = 1.5, len = 21)
    # s2s = noise_gen(f0 = 0.5, len = 21)
    # s3s = noise_gen(f0 = 0.5, len = 21)
    # noise36s = np.append(s1s, s2s)
    # noise36s = np.append(noise36s, s3s)
    #
    # s1s = noise_gen(f0 = 2, len = 21)
    # s2s = noise_gen(f0 = 2, len = 21)
    # s3s = noise_gen(f0 = 2, len = 21)
    # noise38s = np.append(s1s, s2s)
    # noise38s = np.append(noise38s, s3s)
    # plot_wf(sampling_rate = 128, Signal= data36 + 2000 * noise36s + np.random.normal(0,300, size=(8064,)), fft_size = 1024, t_len = 8, name = "data36s" )
    # plot_wf(sampling_rate = 128, Signal= data38 + 2000 * noise38s + np.random.normal(0,100, size=(8064,)), fft_size = 1024, t_len = 8, name = "data38s" )

