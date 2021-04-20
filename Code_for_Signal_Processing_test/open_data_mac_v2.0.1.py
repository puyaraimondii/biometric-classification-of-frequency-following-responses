#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 21:22:58 2018

@author: bruce
"""



import pandas as pd
import os
import numpy as np
from scipy import fftpack
from scipy import signal
import matplotlib.pyplot as plt

pkl_file=pd.read_pickle('/Users/bruce/Documents/uOttawa/Project/audio_brainstem_response/Data_BruceSunMaster_Studies/study2/study2DataFrame.pkl')

# open files in subfolders

def must_open(dirpath, filename):
    if filename.endswith('.txt'):
        return True

def opened_files(*args):
    """generate a sequence of pairs (path to file, opened file)
    in a given directory. Same arguments as os.walk."""
    for dirpath, dirnames, filenames in os.walk(*args):
        for filename in filenames:
            if must_open(dirpath, filename):
                filepath = os.path.join(dirpath, filename)
                yield (filepath, open(filepath, "rU"))


# parameters
sampling_rate = 9606
n = 1024
k = np.arange(n)
T = n/sampling_rate
frq = k/T
freq = frq[range(int(n/2))]



# main loop:
n=1
df_EFR = pd.DataFrame()
mydir = "/Users/bruce/Documents/uOttawa/Project/audio_brainstem_response/Data_BruceSunMaster_Studies/study2/rawdata"
for filepath, file in opened_files(mydir):
    # do something
    _, _, _, _, _, _, _, _, _, _, gender, subject, condition, vowel, sound_level, filename = filepath.split("/")
    temp_EFR = []
    f = open(filepath, 'r')
    data = f.readlines()
    for line in data:
        #Condensation, Rarefaction, EFR, FFR = line.split(",")
        _, _, EFR, _ = line.split(",")
        temp_EFR.append(EFR)
    del temp_EFR[0]
    if n>2:
        n=1
    temp_EFR = np.reshape(temp_EFR, (1,1024))
    label = np.array([subject[1:], gender, condition, vowel, sound_level, n, "EFR"]).reshape(1,7)
    temp_EFR = pd.DataFrame(np.hstack((temp_EFR, label)))
    df_EFR =df_EFR.append(temp_EFR, ignore_index=True)
    print("filepath:", filepath)
    print(label)
    n = n + 1

# not working!!!!
#df_EFR.rename(columns={'1024':"Subjects", '1025':"Sex", "1026":"Condition", "1027":"Vowel", "1028":"Sound Level", "1029":"Num", "1030":"EFR/FFR"})

# set column name
df_EFR.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])

# change the type of data 
df_EFR.iloc[:, 0:1024] = df_EFR.iloc[:, 0:1024].astype(float)
df_EFR.iloc[:, 1024:1025] = df_EFR.iloc[:, 1024:1025].astype(int)
df_EFR.sort_values(by=['1024', '1028', '1029'])
# sort index based on subject condition and sound level
df_EFR_sorted = df_EFR.sort_values(by=['Subject', 'Condition','Sound Level'])


# reset the index
df_EFR_sorted_newindex = df_EFR_sorted.reset_index(drop=True)
# save the final version of .pkl
df_EFR_sorted_newindex.to_pickle('df_EFR.pkl')




'''
# First Graph
plt.figure()
plt.subplot(221)
plt.plot(m_t_a_85_1_cond)
plt.title("Condensation")
plt.ylim(-0.4, 0.4)

plt.subplot(222)
plt.plot(m_t_a_85_1_rare)
plt.title("Rarefaction")
plt.ylim(-0.4, 0.4)

plt.subplot(223)
plt.plot(m_t_a_85_1_EFR)
plt.title("EFR_Signal")
plt.ylim(-0.4, 0.4)

plt.subplot(224)
plt.plot(m_t_a_85_1_FFR)
plt.title("FFR_Signal")
plt.ylim(-0.4, 0.4)
#plt.plot(m_t_a_85_2_EFR)
#plt.legend(["1", "2"])

plt.show()

# comparason of myresult and brian result
plt.figure()
plt.subplot(221)
plt.plot(m_t_a_85_1_EFR)
plt.title("EFR_Signal_Brian")
#plt.ylim(-0.4, 0.4)

plt.subplot(222)
plt.plot(m_t_a_85_1_FFR)
plt.title("FFR_Signal_Brian")
#plt.ylim(-0.4, 0.4)

plt.subplot(223)
plt.plot(m_t_a_85_1_EFRtest)
plt.title("EFR_Signal_Bruce")
#plt.ylim(-0.4, 0.4)

plt.subplot(224)
plt.plot(m_t_a_85_1_FFRtest)
plt.title("FFR_Signal_Brruce")
#plt.ylim(-0.4, 0.4)
plt.show()

'''



'''
# Second Graph
plt.figure()

plt.subplot(221)
plt.plot(freq, s1_m_t_a_85_1_EFR_amplitude_spectrum)
plt.xlim(0,1000)
plt.title("s1_m_t_a_85_1_EFR_amplitude_spectrum(0-1000)")
#plt.xlabel("Frequency")
#plt.ylabel("Amplitude")
#plt.legend(["1", "2"])
plt.grid(True)

plt.subplot(222)
plt.plot(freq, s1_m_re_a_85_1_EFR_amplitude_spectrum)
plt.xlim(0,1000)
plt.grid(True)
plt.title("s1_m_re_a_85_1_EFR_amplitude_spectrum")

plt.subplot(223)
plt.plot(freq, s1_m_t_a_85_2_EFR_amplitude_spectrum)
plt.xlim(0,1000)
plt.grid(True)
plt.title("s1_m_t_a_85_2_EFR_amplitude_spectrum")

plt.subplot(224)
plt.plot(freq, s1_m_re_a_85_2_EFR_amplitude_spectrum)
plt.xlim(0,1000)
plt.grid(True)
plt.title("s1_m_re_a_85_2_EFR_amplitude_spectrum")
plt.show()


# Third Graph
plt.figure()

plt.subplot(221)
plt.plot(freq, s2_f_t_a_85_1_EFR_amplitude_spectrum)
plt.xlim(0,1000)
plt.title("s1_m_t_a_85_1_EFR_amplitude_spectrum(0-1000)")
#plt.xlabel("Frequency")
#plt.ylabel("Amplitude")
#plt.legend(["1", "2"])
plt.grid(True)

plt.subplot(222)
plt.plot(freq, s1_m_re_a_85_1_EFR_amplitude_spectrum)
plt.xlim(0,1000)
plt.grid(True)
plt.title("s1_m_re_a_85_1_EFR_amplitude_spectrum")

plt.subplot(223)
plt.plot(freq, s2_f_t_a_85_2_EFR_amplitude_spectrum)
plt.xlim(0,1000)
plt.grid(True)
plt.title("s1_m_t_a_85_2_EFR_amplitude_spectrum")

plt.subplot(224)
plt.plot(freq, s1_m_re_a_85_2_EFR_amplitude_spectrum)
plt.xlim(0,1000)
plt.grid(True)
plt.title("s1_m_re_a_85_2_EFR_amplitude_spectrum")
plt.show()



# Third Graph
plt.figure()

plt.subplot(221)
plt.plot(corr_s1t1_s1re1)
plt.title("correlation of s1t1 and s1re1")
plt.grid(True)

plt.subplot(222)
plt.plot(corr_s2t1_s1re1)
plt.grid(True)
plt.title("correlation of s2t1 and s1re1")

plt.subplot(223)
plt.plot(corr_s3t1_s1re1)
plt.grid(True)
plt.title("correlation of s3t1 and s1re1")

plt.subplot(224)
plt.plot(corr_s4t1_s1re1)
plt.grid(True)
plt.title("correlation of s4t1 and s1re1")
#plt.show()




#### comparason ####
plt.figure()
temp = pkl_file.iloc[6:7,0:1024]
temp2 = temp.values.T.tolist()
temp2_f = fftpack.fft(temp2)
temp2_f = temp2_f[0:int(len(temp2_f)/2)]
plt.plot(temp2_f)
plt.plot(m_t_a_85_1_EFR_f)
plt.title("Signal_undetrended")
#plt.legend(["1", "2"])
plt.show()

################
'''
