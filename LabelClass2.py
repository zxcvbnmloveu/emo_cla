def onehotencoding2(mode = 0):
    print("Program started"+"\n")
    fout_labels_class = open("./data\label_class_2.dat",'w')
    
    with open('./data\labels_2.dat','r') as f:
        if(mode == 0):
            for val in f:
                if float(val) > 4.5:
                    fout_labels_class.write(str(1) + "\n");
                else:
                    fout_labels_class.write(str(0) + "\n");
        elif(mode == 1):
            for val in f:
                if float(val) < 3:
                    fout_labels_class.write(str(0) + "\n");
                elif float(val) <6:
                    fout_labels_class.write(str(1) + "\n");
                else:
                    fout_labels_class.write(str(2) + "\n");

if __name__ == '__main__':
    onehotencoding2()
