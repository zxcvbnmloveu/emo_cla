import numpy
import numpy as np

a = [1,np.nan,3,1]
b = [4,0,-1,2]

a = np.array(a)
b = np.vstack((a,b))
min = np.min(b,axis=0)
min = np.isnan(min)
print(min)
print(b[:, ~min])




