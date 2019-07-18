import math
import numpy
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import plot
from scipy import stats, signal

nLabel, nTrial, nUser, nChannel, nTime  = 4, 40, 32, 40, 8064
def convertData(channel = 38):
    print("channel " + str(channel) + " Program started"+"\n")
    # adr = "./data/N_C/channel" + str(channel) + "_clear.dat"
    # adrn = "./data/N_C/channel" + str(channel) + "_noise.dat"
    # fout_data = open(adr,'w')
    # fout_data_n = open(adrn,'w')
    fout_data = open("./data/features_MA_noise_less.dat",'w')
    fout_labels0 = open("./data\labels_0.dat",'w')
    fout_labels1 = open("./data\labels_1.dat",'w')
    fout_labels2 = open("./data\labels_2.dat",'w')
    fout_labels3 = open("./data\labels_3.dat",'w')
    t = np.linspace(0,63, num=np.floor(63*128))
    for i in range(32):  #nUser #4, 40, 32, 40, 8064
        if(i%1 == 0):
            if i < 10:
                name = '%0*d' % (2,i+1)
            else:
                name = i+1
        fname = "./data_preprocessed_python\s"+str(name)+".dat"     #C:/Users/lumsys/AnacondaProjects/Emo/
        # noise = 2000*signal.chirp(t = t, f0 = 0.3, t1 = 63, f1 = 3.3, method='linear', phi=np.random.normal(0,100, size=(1,)))
        # noise = np.random.normal(0,3000, size=(8064,))
        # noise = np.zeros((8064,))
        # min = 0.3
        # max = 3.3
        # rand = min + (max-min)*np.random.random()
        # for i in range(500):
        #     fs = min + (max-min)*np.random.random()
        #     # print(fs)
        #     N = 8064
        #     n = [2*math.pi*fs*t/N for t in range(N)]
        #     noise = noise + 10 * np.array([math.sin(i) for i in n]) + np.random.normal(0, 10, size=(8064,))
        noise = numpy.loadtxt('./MA1\MA' + str(i%10+1) + '.txt', delimiter=',')
        f = open(fname, 'rb')
        x = pickle.load(f, encoding='latin1')
        print(fname)
        for tr in range(nTrial):
            start = 0
            if(tr%1 == 0):
                for dat in range(nTime):
                     if dat != 0:
                        if(dat%807 == 0 or dat == 8063):
                        # if(dat%807 == 0 or dat == 8063):
                            # slide_datasn = extract_data((x['data'][tr][channel][start : dat] + noise[start : dat]))
                            slide_datas = extract_data((noise[start : dat]))
                            # slide_datas = extract_data((x['data'][tr][channel][start : dat]+ noise[start : dat]))
                            start = dat
                            # for data in slide_datasn:
                            #     # fout_data_n.write(str(data)+ " ")
                            for data in slide_datas:
                                fout_data.write(str(data)+ " ")

                # datasn = extract_data((x['data'][tr][channel][:] + noise[:]))
                datas = extract_data((x['data'][tr][channel][:] + noise[:]))

                # for data in datasn:
                #     # fout_data_n.write(str(data)+ " ")

                # for data in datas:
                #     fout_data.write(str(data)+ " ")

                # fout_data_n.write(str(tr)+ " ")
                # fout_data_n.write(str(i)+ " ")
                #个性化特征
                # fout_data.write(str(tr)+ " ")
                fout_data.write(str(i)+ " ")

                fout_labels0.write(str(x['labels'][tr][0]) + "\n")
                fout_labels1.write(str(x['labels'][tr][1]) + "\n")
                fout_labels2.write(str(x['labels'][tr][2]) + "\n")
                fout_labels3.write(str(x['labels'][tr][3]) + "\n")
                # fout_data_n.write("\n")#40个特征换行
                fout_data.write("\n")#40个特征换行
    fout_labels0.close()
    fout_labels1.close()
    fout_labels2.close()
    fout_labels3.close()
    # fout_data_n.close()
    fout_data.close()
    print("\n"+"Print Successful")

def extract_data(target_data, a = 0):
    target_res = []

    target_mean = target_data.mean(axis=a)
    target_median = np.median(target_data, axis=a)
    target_maximum = np.max(target_data, axis=a)
    target_minimum = np.min(target_data, axis=a)
    target_std = np.std(target_data, axis=a)
    target_var = np.var(target_data, axis=a)
    target_range = np.ptp(target_data, axis=a)
    target_skew = stats.skew(target_data, axis=a)
    target_kurtosis = stats.kurtosis(target_data, axis=a)

    return [target_mean, target_median, target_maximum, target_minimum, target_std, target_var, target_range, target_skew, target_kurtosis]

if __name__ == '__main__':
    convertData()


