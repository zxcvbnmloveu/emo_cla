import numpy

# x = ["平均系数和","平均系数均值","平均系数标准差","2级细节系数和","2级细节系数均值","2级细节系数标准差","1级细节系数和","1级细节系数均值","1级细节系数标准差",]
x = ["平均系数和","均值","标准差","2级细节系数和","均值","标准差","1级细节系数和","均值","标准差",]
list = []
for i in range(9):
    if (i)%3 == 0:
        list.append("hEOG" + x[i])
    else:
        list.append(x[i])
for i in range(9):
    if (i)%3 == 0:
        list.append("vEOG" + x[i])
    else:
        list.append(x[i])
for i in range(9):
    if (i)%3 == 0:
        list.append("GSR" + x[i])
    else:
        list.append(x[i])
for i in range(9):
    if (i)%3 == 0:
        list.append("PPG" + x[i])
    else:
        list.append(x[i])
print("，".join(list))

# c2
# best indices: (0, 8, 9, 17, 23, 26, 27, 35)
#
# w2
# best indices: (0, 2, 5, 8, 9, 11, 17, 18, 19, 24, 27, 30)
#
# s2
# best indices: (0, 5, 9, 17, 18, 27, 30)
#
#
# c3
# best indices: (0, 8, 9, 17, 18, 21, 26, 27, 32, 35)
#
#
# w3
# best indices: (0, 2, 5, 8, 9, 14, 18, 26, 27, 29)
#
#
# s3
# best indices: (1, 5, 8, 9, 19, 28)


# ll = []
# c2 = [0, 8, 9, 17, 23, 26, 27, 35]
# w2 = [0, 2, 5, 8, 9, 11, 17, 18, 19, 24, 27, 30]
# s2 = [0, 5, 9, 17, 18, 27, 30]
# c3 = [0, 8, 9, 17, 18, 21, 26, 27, 32, 35]
# w3 = [0, 2, 5, 8, 9, 14, 18, 26, 27, 29]
# s3 = [1, 5, 8, 9, 19, 28]
# ll = [c2,w2,s2,c3,w3,s3]

# for index in ll:
#     print("，".join([list[i] for i in index]))

