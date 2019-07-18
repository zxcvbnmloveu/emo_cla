import numpy
import numpy as np
data = np.load("./data/filename.npy")
file_y1 = './data/label_class_0.dat'
file_y2 = './data/label_class_1.dat'
file_y1 = numpy.genfromtxt(file_y1, delimiter=' ')
file_y2 = numpy.genfromtxt(file_y2, delimiter=' ')
#remove nan
d = []
l1 = []
l2 = []
for i, v in enumerate(data):
    if True in numpy.isnan(v):
        pass
        # print(i,v)
    else:
        l1.append(file_y1[i])
        l2.append(file_y2[i])
        d.append(v)
X = numpy.array(d)
y1 = numpy.array(l1)
y2 = numpy.array((l2))
print(X.shape)
print(y1.shape)
print(y2.shape)



