#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:20:10 2018

@author: bruce
"""


import pandas as pd
import numpy as np
from scipy import fftpack
from scipy import signal
import matplotlib.pyplot as plt



def correlation_matrix(corr_mx, cm_title):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #cmap = cm.get_cmap('jet', 30)
    cax = ax1.matshow(corr_mx, cmap='gray')
    #cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    fig.colorbar(cax)
    ax1.grid(False)
    plt.title(cm_title)
    #plt.title('cross correlation of test and retest')
    ylabels=['T1','T2','T3','T4','T6','T7','T8','T9', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T25']
    xlabels=['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
    ax1.set_xticks(np.arange(len(xlabels)))
    ax1.set_yticks(np.arange(len(ylabels)))
    ax1.set_xticklabels(xlabels,fontsize=6)
    ax1.set_yticklabels(ylabels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    #fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()


def correlation_matrix_01(corr_mx, cm_title):
    # find the maximum in each row 

    # input corr_mx is a dataframe
    # need to convert it into a array first
    #otherwise it is not working
    temp = np.asarray(corr_mx)
    output = (temp == temp.max(axis=1)[:,None]) # along rows

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #cmap = cm.get_cmap('jet', 30)
    cax = ax1.matshow(output, cmap='gray')
    #cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    fig.colorbar(cax)
    ax1.grid(False)
    plt.title(cm_title)
    ylabels=['T1','T2','T3','T4','T6','T7','T8','T9', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T25']
    xlabels=['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
    ax1.set_xticks(np.arange(len(xlabels)))
    ax1.set_yticks(np.arange(len(ylabels)))
    ax1.set_xticklabels(xlabels,fontsize=6)
    ax1.set_yticklabels(ylabels,fontsize=6)
    plt.show()
    
    

f_dB = lambda x : 20 * np.log10(np.abs(x)) 


# import the pkl file
#pkl_file=pd.read_pickle('/Users/bruce/Documents/uOttawa/Project/audio_brainstem_response/Data_BruceSunMaster_Studies/study2/study2DataFrame.pkl')
df_EFR=pd.read_pickle('/home/bruce/Dropbox/Project/4.Code for Linux/df_EFR.pkl')

# remove DC offset
df_EFR_detrend = pd.DataFrame()
for i in range(1408):
    # combine next two rows later
    df_EFR_detrend_data = pd.DataFrame(signal.detrend(df_EFR.iloc[i: i+1, 0:1024], type='constant').reshape(1,1024))
    df_EFR_label_temp = pd.DataFrame(df_EFR.iloc[i, 1024:1031].values.reshape(1,7))
    df_EFR_detrend = df_EFR_detrend.append(pd.concat([df_EFR_detrend_data, df_EFR_label_temp], axis=1, ignore_index=True))
# set the title of columns
df_EFR_detrend.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_EFR_detrend = df_EFR_detrend.reset_index(drop=True)
df_EFR = df_EFR_detrend


# Time domain


# Define window function
win_kaiser = signal.kaiser(1024, beta=14)
win_hamming = signal.hamming(1024)

# average the df_EFR
df_EFR_avg = pd.DataFrame()
df_EFR_avg_win = pd.DataFrame()
# average test1 and test2
for i in range(704):
    # combine next two rows later
    df_EFR_avg_t = pd.DataFrame(df_EFR.iloc[2*i: 2*i+2, 0:1024].mean(axis=0).values.reshape(1,1024)) # average those two rows
    # implement the window function
    df_EFR_avg_t_win = pd.DataFrame((df_EFR_avg_t.iloc[0,:] * win_hamming).values.reshape(1,1024))
    
    df_EFR_label = pd.DataFrame(df_EFR.iloc[2*i, 1024:1031].values.reshape(1,7))
    df_EFR_avg = df_EFR_avg.append(pd.concat([df_EFR_avg_t, df_EFR_label], axis=1, ignore_index=True))
    df_EFR_avg_win = df_EFR_avg.append(pd.concat([df_EFR_avg_t_win, df_EFR_label], axis=1, ignore_index=True))

# set the title of columns
df_EFR_avg.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_EFR_avg = df_EFR_avg.sort_values(by=["Condition", "Subject"])
df_EFR_avg = df_EFR_avg.reset_index(drop=True)

df_EFR_avg_win.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_EFR_avg_win = df_EFR_avg_win.sort_values(by=["Condition", "Subject"])
df_EFR_avg_win = df_EFR_avg_win.reset_index(drop=True)

# average all the subjects , test and retest and keep one sound levels
# filter by 'a vowel and 85Db'
df_EFR_avg_sorted = df_EFR_avg.sort_values(by=["Sound Level", "Vowel", "Subject", "Condition"])
df_EFR_avg_sorted = df_EFR_avg_sorted.reset_index(drop=True)
df_EFR_avg_win_sorted = df_EFR_avg_win.sort_values(by=["Sound Level", "Vowel", "Subject", "Condition"])
df_EFR_avg_win_sorted = df_EFR_avg_win_sorted.reset_index(drop=True)
# filter55 65 75 sound levels and keep 85dB
# keep vowel condition and subject
df_EFR_avg_85 = pd.DataFrame(df_EFR_avg_sorted.iloc[528:, :])
df_EFR_avg_85 = df_EFR_avg_85.reset_index(drop=True)
df_EFR_avg_win_85 = pd.DataFrame(df_EFR_avg_win_sorted.iloc[528:, :])
df_EFR_avg_win_85 = df_EFR_avg_win_85.reset_index(drop=True)



# average subjects, conditions
df_EFR_avg_85_aenu = pd.DataFrame()
df_EFR_avg_win_85_aenu = pd.DataFrame()
for i in range(4):
    # combine next two rows later
    df_EFR_avg_t = pd.DataFrame(df_EFR_avg_85.iloc[44*i: 44*i+44, 0:1024].mean(axis=0).values.reshape(1,1024)) # average those two rows
    df_EFR_avg_label = pd.DataFrame(df_EFR_avg_85.iloc[44*i, 1024:1031].values.reshape(1,7))
    temp = pd.concat([df_EFR_avg_t, df_EFR_avg_label], axis=1, ignore_index=True)
    df_EFR_avg_85_aenu = df_EFR_avg_85_aenu.append(temp, ignore_index=True)
    
    df_EFR_avg_win_t = pd.DataFrame(df_EFR_avg_win_85.iloc[44*i: 44*i+44, 0:1024].mean(axis=0).values.reshape(1,1024)) # average those two rows
    df_EFR_avg_win_label = pd.DataFrame(df_EFR_avg_win_85.iloc[44*i, 1024:1031].values.reshape(1,7))
    temp_win = pd.concat([df_EFR_avg_win_t, df_EFR_avg_win_label], axis=1, ignore_index=True)
    df_EFR_avg_win_85_aenu = df_EFR_avg_win_85_aenu.append(temp_win, ignore_index=True)

# set the title of columns
df_EFR_avg_85_aenu.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_EFR_avg_85_aenu = df_EFR_avg_85_aenu.sort_values(by=["Condition", "Subject"])
df_EFR_avg = df_EFR_avg.reset_index(drop=True)

df_EFR_avg_win_85_aenu.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_EFR_avg_win_85_aenu = df_EFR_avg_win_85_aenu.sort_values(by=["Condition", "Subject"])
df_EFR_avg_win = df_EFR_avg_win.reset_index(drop=True)


 
##################################################
# Frequency Domain   

# parameters
sampling_rate = 9606 # fs
# sampling_rate = 9596.623
n = 1024
k = np.arange(n)
T = n/sampling_rate # time of signal
frq = k/T
freq = frq[range(int(n/2))]

n2 = 96060
k2 = np.arange(n2)
T2 = n2/sampling_rate
frq2 = k2/T2
freq2 = frq2[range(int(n2/2))]



# zero padding

# for df_EFR
df_EFR_data = df_EFR.iloc[:, :1024]
df_EFR_label = df_EFR.iloc[:, 1024:]


# for df_EFR_avg_85
df_EFR_avg_85_data = df_EFR_avg_85.iloc[:, :1024]
df_EFR_avg_85_label = df_EFR_avg_85.iloc[:, 1024:]
df_EFR_avg_85_mid = pd.DataFrame(np.zeros((176, 95036)))
df_EFR_avg_85_withzero = pd.concat([df_EFR_avg_85_data, df_EFR_avg_85_mid, df_EFR_avg_85_label], axis=1)
df_EFR_avg_85_withzero.columns = np.append(np.arange(96060), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])

# df_EFR_avg_win_85
df_EFR_avg_win_85_data = df_EFR_avg_win_85.iloc[:, :1024]
df_EFR_avg_win_85_label = df_EFR_avg_win_85.iloc[:, 1024:]
df_EFR_avg_win_85_mid = pd.DataFrame(np.zeros((176, 95036)))
df_EFR_avg_win_85_withzero = pd.concat([df_EFR_avg_win_85_data, df_EFR_avg_win_85_mid, df_EFR_avg_win_85_label], axis=1)
df_EFR_avg_win_85_withzero.columns = np.append(np.arange(96060), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])

# for df_EFR_avg_85
df_EFR_avg_85_aenu_data = df_EFR_avg_85_aenu.iloc[:, :1024]
df_EFR_avg_85_aenu_label = df_EFR_avg_85_aenu.iloc[:, 1024:]

# for df_EFR_avg_win_85
df_EFR_avg_win_85_aenu_data = df_EFR_avg_win_85_aenu.iloc[:, :1024]
df_EFR_avg_win_85_aenu_label = df_EFR_avg_win_85_aenu.iloc[:, 1024:]



# Calculate the Amplitude Spectrum


# for df_EFR_avg_aenu_withzero
df_win_85_aenu_as= pd.DataFrame()
df_win_85_aenu_as7= pd.DataFrame()
for i in range(4):
    # y = 2/N * np.abs (freq_data [0:np.int (N/2)])
    temp_aenu_as = 2/n2 *np.abs((np.fft.fft(df_EFR_avg_win_85_aenu_data.iloc[i, :], n=96060))[range(int(n2/2))])
    
    df_win_85_aenu_as = df_win_85_aenu_as.append(pd.DataFrame(temp_aenu_as.reshape(1,48030)), ignore_index = True)
    df_win_85_aenu_as7 = df_win_85_aenu_as7.append(pd.DataFrame(np.array([temp_aenu_as[1000], temp_aenu_as[2000], temp_aenu_as[3000], temp_aenu_as[4000], \
                                                                  temp_aenu_as[5000], temp_aenu_as[6000], temp_aenu_as[7000]]).reshape(1,7)), ignore_index = True)

df_win_85_aenu_as = pd.concat([df_win_85_aenu_as, df_EFR_avg_85_aenu_label], axis=1) # add labels on it
df_win_85_aenu_as7 = pd.concat([df_win_85_aenu_as7, df_EFR_avg_85_aenu_label], axis=1) # add labels on it



# plot the figure

# grand mean aenu vowels 

plt.figure()
plt.subplot(1, 2, 1)
x_label = np.arange(0, 100, 0.09765625)
plt.plot(x_label, np.asarray(df_EFR_avg_85_aenu.iloc[0,:1024]))
plt.plot(x_label, np.asarray(df_EFR_avg_85_aenu.iloc[1,:1024]))
plt.plot(x_label, np.asarray(df_EFR_avg_85_aenu.iloc[2,:1024]))
plt.plot(x_label, np.asarray(df_EFR_avg_85_aenu.iloc[3,:1024]))
plt.title('grand mean 85dB 4 vowel envelope FFRs in time domain')
plt.legend(('a vowel', 'e vowel', 'n vowel', 'u vowel'), loc='upper right')
plt.xlabel('Time (ms)')

plt.subplot(1, 2, 2)
#markers = [100, 200, 300, 400, 500, 600, 700] # which corresponds to 100 200....700Hz in frequency domain
x_label = np.arange(0, 4803, 0.1)
plt.plot(x_label, df_win_85_aenu_as.iloc[0, :48030], '-')
plt.plot(x_label, df_win_85_aenu_as.iloc[1, :48030], '-')
plt.plot(x_label, df_win_85_aenu_as.iloc[2, :48030], '-')
plt.plot(x_label, df_win_85_aenu_as.iloc[3, :48030], '-')
plt.title('grand-mean 85dB 4 vowel envelope FFRs in frequency domain')
plt.legend(('a vowel', 'e vowel', 'n vowel', 'u vowel'), loc='upper right')
plt.xlabel('Frequency(Hz)')
plt.xlim(0,1300)
plt.show()

# plot dB
'''
plt.figure()
plt.plot(np.asarray(df_EFR_avg_85_aenu.iloc[0,:1024].apply(f_dB)))
plt.plot(np.asarray(df_EFR_avg_85_aenu.iloc[1,:1024].apply(f_dB)))
plt.plot(np.asarray(df_EFR_avg_85_aenu.iloc[2,:1024].apply(f_dB)))
plt.plot(np.asarray(df_EFR_avg_85_aenu.iloc[3,:1024].apply(f_dB)))
plt.title('grand mean 4 vowel signal in time domain(dB)')
plt.ylabel('dB')
plt.show()
'''



#print ("max of a:", np.argmax(df_85_aenu_as.iloc[0, :48030])) # 999
#print ("max of e:", np.argmax(df_85_aenu_as.iloc[1, :48030])) # 1004
#print ("max of n:", np.argmax(df_85_aenu_as.iloc[2, :48030])) # 1002
#print ("max of u:", np.argmax(df_85_aenu_as.iloc[3, :48030])) # 991

'''
plt.figure()
#markers = [100, 200, 300, 400, 500, 600, 700] # which corresponds to 100 200....700Hz in frequency domain
plt.plot(signal.resample(df_85_aenu_as.iloc[0, :48030], 48078), '-')
plt.plot(signal.resample(df_85_aenu_as.iloc[1, :48030], 47839), '-')
plt.plot(signal.resample(df_85_aenu_as.iloc[2, :48030], 47934), '-')
plt.plot(signal.resample(df_85_aenu_as.iloc[3, :48030], 48466), '-')
plt.title('resampled grand mean 85dB 4 vowel signal in frequency domain')
plt.xlim(0,10000)
plt.legend(('a', 'e', 'n', 'u'), loc='upper right')
plt.show()
'''

# plot dB
'''
plt.figure()
#markers = [100, 200, 300, 400, 500, 600, 700] # which corresponds to 100 200....700Hz in frequency domain
plt.plot(df_85_aenu_as.iloc[0, :48030].apply(f_dB), '-')
plt.plot(df_85_aenu_as.iloc[1, :48030].apply(f_dB), '-')
plt.plot(df_85_aenu_as.iloc[2, :48030].apply(f_dB), '-')
plt.plot(df_85_aenu_as.iloc[3, :48030].apply(f_dB), '-')
plt.title('grand mean 4 vowel signal in frequency domain(dB)')
plt.xlim(0,10000)
plt.xlabel('Frequency(Hz)')
plt.ylabel('dB')
plt.show()
'''

# figure a e n u in 1 plot
fig, axes = plt.subplots(4,2, sharex=False)
x_label_time = np.arange(0, 100, 0.09765625)
x_label_freq = np.arange(0, 4803, 0.1)
#markers = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000] # which corresponds to 100 200....700Hz in frequency domain

# a vowel
axes[0, 0].plot(x_label_time, np.asarray(df_EFR_avg_85_aenu.iloc[0,:1024]))
axes[0, 0].set_xticks([])
axes[0, 0].set_ylabel(r'$Amplitude\  (\mu V) $')
axes[0, 1].plot(x_label_freq, df_win_85_aenu_as.iloc[0, :48030], label="a vowel") 
axes[0, 1].set_xlim(0, 1300)
axes[0, 1].legend(loc="upper right")

# e vowel
axes[1, 0].plot(x_label_time, np.asarray(df_EFR_avg_85_aenu.iloc[1,:1024]))
axes[1, 0].set_xticks([])
axes[1, 0].set_ylabel(r'$Amplitude\  (\mu V) $')
axes[1, 1].plot(x_label_freq, df_win_85_aenu_as.iloc[1, :48030], label="e vowel")  
axes[1, 1].set_xlim(0, 1300)
axes[1, 1].legend(loc="upper right")

# n vowel
axes[2, 0].plot(x_label_time, np.asarray(df_EFR_avg_85_aenu.iloc[2,:1024]))
axes[2, 0].set_xticks([])
axes[2, 0].set_ylabel(r'$Amplitude\  (\mu V) $')
axes[2, 1].plot(x_label_freq, df_win_85_aenu_as.iloc[2, :48030], label="n vowel")  
axes[2, 1].set_xlim(0, 1300)
axes[2, 1].legend(loc="upper right")

# u vowel
axes[3, 0].plot(x_label_time, np.asarray(df_EFR_avg_85_aenu.iloc[3,:1024]))
axes[3, 0].set_xlabel('Time (ms)')
axes[3, 0].set_ylabel(r'$Amplitude\  (\mu V) $')
axes[3, 1].plot(x_label_freq, df_win_85_aenu_as.iloc[3, :48030], label="u vowel")
axes[3, 1].set_xlim(0, 1300)
axes[3, 1].set_xlabel('Frequency(Hz)')
axes[3, 1].legend(loc="upper right")

fig.suptitle('grand mean 85dB 4 vowel envelope FFRs in time and frequency domain')
plt.show()


# Correlation Matrix

'''
# corr_EFR_a_85
#plt.subplot(1,3,1)
plt.matshow(corr_EFR_a_85_test)# cmap=plt.cm.gray
plt.title('cross correlation of test subject')
plt.colorbar() # show the color bar on the right side of the figure

#plt.subplot(1,3,2)
plt.matshow(corr_EFR_a_85_retest) # cmap=plt.cm.gray
plt.title('cross correlation of retest subject')
plt.colorbar() # show the color bar on the right side of the figure

#plt.subplot(1,3,3)
plt.matshow(corr_EFR_a_85_r_t) # cmap=plt.cm.gray
plt.title('cross correlation of retest-test')
plt.colorbar() # show the color bar on the right side of the figure


plt.matshow(corr_EFR_a_85_r_t_part)
plt.title('cross correlation of test and retest')
plt.colorbar()
'''
 


#correlation_matrix(corr_EFR_a_85_r_t_part, 'a_vowel_85Db cross correlation of test and retest')
#correlation_matrix_01(corr_EFR_a_85_r_t_part, 'a_vowel_85Db cross correlation of test and retest')



'''
# original test

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.matshow(corr_EFR_a_85_r_t_part, cmap='gray') # cmap=plt.cm.gray
plt.title('cross correlation of test and retest')
plt.colorbar() # show the color bar on the right side of the figure
ax1.grid(False)
ylabels=['T1','T2','T3','T4','T6','T7','T8','T9', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T25']
xlabels=['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
ax1.set_xticks(np.arange(len(xlabels)))
ax1.set_yticks(np.arange(len(ylabels)))
ax1.set_xticklabels(xlabels,fontsize=6)
ax1.set_yticklabels(ylabels,fontsize=6)
'''



'''
#plt.subplot(1,3,1)
plt.matshow(corr_as_test)# cmap=plt.cm.gray
plt.title('cross correlation of test subject')
plt.colorbar() # show the color bar on the right side of the figure

#plt.subplot(1,3,2)
plt.matshow(corr_as_retest) # cmap=plt.cm.gray
plt.title('cross correlation of retest subject')
plt.colorbar() # show the color bar on the right side of the figure

#plt.subplot(1,3,3)
plt.matshow(corr_as_t_r) # cmap=plt.cm.gray
plt.title('cross correlation of test and retest')
plt.colorbar() # show the color bar on the right side of the figure


plt.matshow(corr_as_norm_test)# cmap=plt.cm.gray
plt.title('auto correlation of normalized test subject')
plt.colorbar() # show the color bar on the right side of the figure

#plt.subplot(1,3,2)
plt.matshow(corr_as_norm_retest) # cmap=plt.cm.gray
plt.title('auto correlation of normalized retest subject')
plt.colorbar() # show the color bar on the right side of the figure

#plt.subplot(1,3,3)
plt.matshow(corr_as_norm_t_r) # cmap=plt.cm.gray
plt.title('corss correlation of normalized test and retest')
plt.colorbar() # show the color bar on the right side of the figure
'''

