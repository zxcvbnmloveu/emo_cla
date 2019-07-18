import numpy as np
import scipy.io as sio
import os
import sys
import random
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import mne
import pyedflib as edf
import random
from scipy import signal
import math
import shutil
from sklearn import preprocessing

DEAP_DATA_DIR_ORIGINAL = "./data_original"
DEAP_DATA_DIR_PREPROCESSED = "./data_preprocessed_python"
AMIGOS_DATA_DIR_ORIGINAL = "D:\AGH\Magisterka\Project\Datasets\AMIGOS\Original"
AMIGOS_DATA_DIR_PREPROCESSED = "D:\AGH\Magisterka\Project\Datasets\AMIGOS\Preprocessed"

def unpickleFile(filename):
    return pickle.load(open(DEAP_DATA_DIR_PREPROCESSED + "\\" + filename, 'rb'), encoding='latin1')

def createDirIfNotExist(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

# DEAP_DATA_ORIGINAL
def processOriginalDEAPDataFile():
    print("processOriginalDEAPDataFile started:")
    no = 0
    ratings = pd.read_csv("./metadata_csv\participant_ratings.csv", header=None)
    for person_index in range(1,33):
        print(DEAP_DATA_DIR_ORIGINAL + "\\s%s.bdf" % str(person_index).zfill(2))
        f = edf.EdfReader(DEAP_DATA_DIR_ORIGINAL + "\\s%s.bdf" % str(person_index).zfill(2))
        # gsrData = f.readSignal(40)
        gsrData = f.readSignal(45)
        markerData = f.readSignal(47) + 6815744.0
        start_index = 0
        end_index = 0
        video_id = 1
        for i in range(len(gsrData)):
            createDirIfNotExist(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\features")
            createDirIfNotExist(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\labels")
            createDirIfNotExist(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\participant_ratings")
            if((i - start_index) > 200 and abs(markerData[i] - 4.0) < 0.1):
                start_index = i
            if((i - end_index) > 200 and abs(markerData[i] - 5.0) < 0.1):
                end_index = i
                if(start_index < end_index):
                    #print("%d, %d" % (start_index, end_index))
                    print("no = " + str(no))
                    np.savetxt(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\features\\" + str(no) + ".csv", gsrData[start_index:end_index:4], fmt="%f", delimiter=",")

                    np.savetxt(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\labels\\" + str(no) + ".csv", [getVAClass(ratings.iat[video_id + (40*(person_index -1)), 5], ratings.iat[video_id + (40*(person_index -1)), 4])], fmt="%d", delimiter=",")
                    # np.savetxt(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\participant_ratings\\" + str(no) + ".csv", ratings.iloc[[video_id + (40*(person_index -1))]], fmt="%.2f", delimiter=",")
                    no += 1
                    video_id += 1
        f._close()

def getAdditionalDEAPOriginalClasses():
    no = 0
    ratings = pd.read_csv("D:\AGH\Magisterka\Project\Datasets\DEAP\metadata\metadata_csv\participant_ratings.csv", header=None)
    createDirIfNotExist(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\labels")
    createDirIfNotExist(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\participant_ratings")
    createDirIfNotExist(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\labels_valence")
    createDirIfNotExist(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\labels_arousal")
    createDirIfNotExist(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\labels_dominance")
    createDirIfNotExist(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\labels_liking")
    for person_index in range(1,33):
        for video_id in range(40):
            np.savetxt(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\labels\\" + str(no) + ".csv", [getVAClass(ratings.iat[video_id + (40*(person_index -1)), 5], ratings.iat[video_id + (40*(person_index -1)), 4])], fmt="%d", delimiter=",")
            np.savetxt(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\participant_ratings\\" + str(no) + ".csv", ratings.iloc[[video_id + (40*(person_index - 1))]], fmt="%.2f", delimiter=",")
            np.savetxt(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\labels_valence\\" + str(no) + ".csv", [getBinaryClass(ratings.iat[video_id + (40*(person_index -1)), 4])], fmt="%d", delimiter=",")
            np.savetxt(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\labels_arousal\\" + str(no) + ".csv", [getBinaryClass(ratings.iat[video_id + (40*(person_index -1)), 5])], fmt="%d", delimiter=",")
            np.savetxt(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\labels_dominance\\" + str(no) + ".csv", [getBinaryClass(ratings.iat[video_id + (40*(person_index -1)), 6])], fmt="%d", delimiter=",")
            np.savetxt(DEAP_DATA_DIR_ORIGINAL + "\\new_edition\\labels_liking\\" + str(no) + ".csv", [getBinaryClass(ratings.iat[video_id + (40*(person_index -1)), 7])], fmt="%d", delimiter=",")
            no += 1


def getBinaryClass(value):
    if(float(value) < 5.):
        return 0
    else:
        return 1

# DEAP_DATA_PREPROCESSED
def processPreprocessedDEAPDataFile():
    OUT_PATH = "./data\csv_features"
    test_index = 0
    for person_index in range(1,33):
        experimentData = unpickleFile("s%s.dat" % str(person_index).zfill(2))
        print(DEAP_DATA_DIR_ORIGINAL + "\\s%s.bdf" % str(person_index).zfill(2))
        labels = experimentData['labels']   #40 x 4(video x label)
        gsrData = experimentData['data']
        data = experimentData['data']       #40 x 40 x 8064(video x channel x data)

        for index in range(0,40):
            createDirIfNotExist(OUT_PATH + "\\features")
            createDirIfNotExist(OUT_PATH + "\\labels")
            createDirIfNotExist(OUT_PATH + "\\participant_ratings")
            vaClass = getVAClass(labels[index][1], labels[index][0])
            if(vaClass < 4):
                if not os.path.exists(OUT_PATH + "\\features"):
                    os.makedirs(OUT_PATH + "\\features")
                if not os.path.exists(OUT_PATH + "\\labels"):
                    os.makedirs(OUT_PATH + "\\labels")
                b, a = signal.butter(5, 0.01, 'low')
                # filtered_gsr[100::25]
                filtered_gsr = signal.filtfilt(b, a, gsrData[index][36])
                # np.savetxt(OUT_PATH + "\\features\\" + str(test_index) + ".csv", filtered_gsr[:], fmt="%7.2f", delimiter=",")
                np.savetxt(OUT_PATH + "\\features\\" + str(test_index) + ".csv", gsrData[index][38][:], fmt="%7.2f", delimiter=",")
                np.savetxt(OUT_PATH + "\\labels\\" + str(test_index) + ".csv", [vaClass], fmt="%d", delimiter=",")
                np.savetxt(OUT_PATH + "\\participant_ratings\\" + str(test_index) + ".csv", [labels[index]], fmt="%f", delimiter=",")
                test_index += 1

def getAdditionalDEAPPreprocessedClasses():
    no = 0
    createDirIfNotExist(DEAP_DATA_DIR_PREPROCESSED + "\\new_edition\\labels_valence")
    createDirIfNotExist(DEAP_DATA_DIR_PREPROCESSED + "\\new_edition\\labels_arousal")
    createDirIfNotExist(DEAP_DATA_DIR_PREPROCESSED + "\\new_edition\\labels_dominance")
    createDirIfNotExist(DEAP_DATA_DIR_PREPROCESSED + "\\new_edition\\labels_liking")
    for person_index in range(1,33):
        experimentData = unpickleFile("s%s.dat" % str(person_index).zfill(2))
        labels = experimentData['labels']   #40 x 4(video x label)
        for video_id in range(40):
            np.savetxt(DEAP_DATA_DIR_PREPROCESSED + "\\new_edition\\labels_valence\\" + str(no) + ".csv", [getBinaryClass(labels[video_id][0])], fmt="%d", delimiter=",")
            np.savetxt(DEAP_DATA_DIR_PREPROCESSED + "\\new_edition\\labels_arousal\\" + str(no) + ".csv", [getBinaryClass(labels[video_id][1])], fmt="%d", delimiter=",")
            np.savetxt(DEAP_DATA_DIR_PREPROCESSED + "\\new_edition\\labels_dominance\\" + str(no) + ".csv", [getBinaryClass(labels[video_id][2])], fmt="%d", delimiter=",")
            np.savetxt(DEAP_DATA_DIR_PREPROCESSED + "\\new_edition\\labels_liking\\" + str(no) + ".csv", [getBinaryClass(labels[video_id][3])], fmt="%d", delimiter=",")
            no += 1


def getVAClass(arousal, valence):
    '''if(arousal >= 6. and valence >= 6.):
        return 0 #"HAHV"
    elif(arousal >= 6. and valence < 4.):
        return 1 #"HALV"
    elif(arousal < 4. and valence >= 6.):
        return 2 #"LAHV"
    elif(arousal < 4. and valence < 4.):
        return 3 #"LALV"
    return 4'''
    arousal = float(arousal)
    valence = float(valence)
    if(arousal >= 5. and valence >= 5.):
        return 0 #"HAHV"
    elif(arousal >= 5. and valence < 5.):
        return 1 #"HALV"
    elif(arousal < 5. and valence >= 5.):
        return 2 #"LAHV"
    elif(arousal < 5. and valence < 5.):
        return 3 #"LALV"

processOriginalDEAPDataFile()
# processPreprocessedDEAPDataFile()
