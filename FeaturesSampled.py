import pickle

nLabel, nTrial, nUser, nChannel, nTime  = 4, 40, 32, 40, 8064

def sampleFeatures():
    print("Program started"+"\n")
    fout_data = open('./data/features_sampled.dat','w')
    for i in range(32):   #nUser  #4, 40, 32, 40, 8064
        if(i%1 == 0):
            if i < 10:			
                name = '%0*d' % (2,i+1)
            else:
                name = i+1		
            fname = "./data_preprocessed_python\s"+str(name)+".dat"		     # C:/Users/lumsys/AnacondaProjects/Emo/
            x = pickle.load(open(fname, 'rb'), encoding='latin1')		
            print(fname)		
            for tr in range(nTrial):    #40trial x 40 channel x 8064point
                if(tr%1 == 0):
                    for dat in range(nTime):					
                        if(dat%64 == 0):			# for each 64 point
                            for ch in range(nChannel):							
                                fout_data.write(str(ch+1) + " ");
                                fout_data.write(str(x['data'][tr][ch][dat]) + " ");
                fout_data.write("\n");
    fout_data.close()
    print('Completed')

if __name__ == '__main__':
    sampleFeatures()
