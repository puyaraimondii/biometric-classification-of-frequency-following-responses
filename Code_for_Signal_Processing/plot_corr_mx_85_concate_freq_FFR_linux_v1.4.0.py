#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 17:02:59 2018

@author: bruce
"""

import pandas as pd
import numpy as np
from scipy import fftpack
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap



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
    
    # show digit in matrix
    corr_mx_array = np.asarray(corr_mx)
    for i in range(22):
        for j in range(22):
            c = corr_mx_array[j,i]
            ax1.text(i, j, round(c,2), va='center', ha='center')
            
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
    cs = ax1.matshow(output, cmap='gray')
    #cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    fig.colorbar(cs)
    ax1.grid(False)
    plt.title(cm_title)
    ylabels=['T1','T2','T3','T4','T6','T7','T8','T9', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T25']
    xlabels=['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
    ax1.set_xticks(np.arange(len(xlabels)))
    ax1.set_yticks(np.arange(len(ylabels)))
    ax1.set_xticklabels(xlabels,fontsize=6)
    ax1.set_yticklabels(ylabels,fontsize=6)
    plt.show()
    

def correlation_matrix_rank(corr_mx, cm_title):
    temp = corr_mx
    #output = (temp == temp.max(axis=1)[:,None]) # along row
    output = temp.rank(axis=1, ascending=False)
    fig, ax1 = plt.subplots()
    im1 = ax1.matshow(output, cmap=plt.cm.Wistia)
    #cs = ax1.matshow(output)
    fig.colorbar(im1)
    ax1.grid(False)
    ylabels=['T1','T2','T3','T4','T6','T7','T8','T9', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T25']
    xlabels=['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
    ax1.set_xticks(np.arange(len(xlabels)))
    ax1.set_yticks(np.arange(len(ylabels)))
    ax1.set_xticklabels(xlabels,fontsize=6)
    ax1.set_yticklabels(ylabels,fontsize=6)
    plt.title(cm_title)
    # show digit in matrix
    output = np.asarray(output)
    for i in range(22):
        for j in range(22):
            c = output[j,i]
            ax1.text(i, j, int(c), va='center', ha='center')
    plt.show()


def correlation_matrix_comb(corr_mx, cm_title):  
    fig, (ax2, ax3) = plt.subplots(1, 2)
    
    ylabels=['T1','T2','T3','T4','T6','T7','T8','T9', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T25']
    xlabels=['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
    
    '''
    # graph 1 grayscale
    im1 = ax1.matshow(corr_mx, cmap='gray')
    # colorbar need numpy version 1.13.1
    #fig.colorbar(im1, ax=ax1)
    ax1.grid(False)
    ax1.set_title(cm_title)
    ax1.set_xticks(np.arange(len(xlabels)))
    ax1.set_yticks(np.arange(len(ylabels)))
    ax1.set_xticklabels(xlabels,fontsize=6)
    ax1.set_yticklabels(ylabels,fontsize=6)
    # show digit in matrix
    corr_mx_array = np.asarray(corr_mx)
    for i in range(22):
        for j in range(22):
            c = corr_mx_array[j,i]
            ax1.text(i, j, round(c,2), va='center', ha='center')
    '''
    
    # graph 2 yellowscale
    corr_mx_rank = corr_mx.rank(axis=1, ascending=False)
    cmap_grey = LinearSegmentedColormap.from_list('mycmap', ['white', 'black'])
    im2 = ax2.matshow(corr_mx, cmap='viridis')
    # colorbar need numpy version 1.13.1
    fig.colorbar(im2, ax=ax2)
    ax2.grid(False)
    ax2.set_title(cm_title)
    ax2.set_xticks(np.arange(len(xlabels)))
    ax2.set_yticks(np.arange(len(ylabels)))
    ax2.set_xticklabels(xlabels,fontsize=6)
    ax2.set_yticklabels(ylabels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    # show digit in matrix
    corr_mx_rank = np.asarray(corr_mx_rank)
    for i in range(22):
        for j in range(22):
            c = corr_mx_rank[j,i]
            ax2.text(i, j, int(c), va='center', ha='center')
    
    # graph 3
    # find the maximum in each row 
    # input corr_mx is a dataframe
    # need to convert it into a array first
    #otherwise it is not working
    temp = np.asarray(corr_mx)
    output = (temp == temp.max(axis=1)[:,None]) # along rows
    im3 = ax3.matshow(output, cmap='gray')
    # colorbar need numpy version 1.13.1
    #fig.colorbar(im3, ax=ax3)
    ax3.grid(False)
    ax3.set_title(cm_title)
    ax3.set_xticks(np.arange(len(xlabels)))
    ax3.set_yticks(np.arange(len(ylabels)))
    ax3.set_xticklabels(xlabels,fontsize=6)
    ax3.set_yticklabels(ylabels,fontsize=6)
    plt.show()


def correlation_matrix_tt_01(corr_mx, cm_title):
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
    xlabels=['T1','T2','T3','T4','T6','T7','T8','T9', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T25']
    ax1.set_xticks(np.arange(len(xlabels)))
    ax1.set_yticks(np.arange(len(ylabels)))
    ax1.set_xticklabels(xlabels,fontsize=6)
    ax1.set_yticklabels(ylabels,fontsize=6)
    plt.show()

def correlation_matrix_rr_01(corr_mx, cm_title):
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
    ylabels=['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
    xlabels=['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
    ax1.set_xticks(np.arange(len(xlabels)))
    ax1.set_yticks(np.arange(len(ylabels)))
    ax1.set_xticklabels(xlabels,fontsize=6)
    ax1.set_yticklabels(ylabels,fontsize=6)
    plt.show()  


# shrink value for correlation matrix
# in order to use colormap -> 10 scale
def shrink_value_03_1(corr_in1):
    corr_out1 = corr_in1
    for i in range (22):
        for j in range(22):
            if corr_in1.iloc[i, j] < 0.3:
                corr_out1.iloc[i, j] = 0.3
    return corr_out1


def shrink_value_05_1(corr_in2):
    corr_out2 = corr_in2
    for i in range (22):
        for j in range(22):
            if corr_in2.iloc[i, j] < 0.5:
                corr_out2.iloc[i, j] = 0.5
    return corr_out2


# import the pkl file
#pkl_file=pd.read_pickle('/Users/bruce/Documents/uOttawa/Project/audio_brainstem_response/Data_BruceSunMaster_Studies/study2/study2DataFrame.pkl')
df_FFR=pd.read_pickle('/home/bruce/Dropbox/Project/4.Code for Linux/df_FFR.pkl')

# remove DC offset
df_FFR_detrend = pd.DataFrame()
for i in range(1408):
    # combine next two rows later
    df_FFR_detrend_data = pd.DataFrame(signal.detrend(df_FFR.iloc[i: i+1, 0:1024], type='constant').reshape(1,1024))
    df_FFR_label = pd.DataFrame(df_FFR.iloc[i, 1024:1031].values.reshape(1,7))
    df_FFR_detrend = df_FFR_detrend.append(pd.concat([df_FFR_detrend_data, df_FFR_label], axis=1, ignore_index=True))
# set the title of columns
df_FFR_detrend.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_FFR_detrend = df_FFR_detrend.reset_index(drop=True)
df_FFR = df_FFR_detrend


# Define window function
win_kaiser = signal.kaiser(1024, beta=14)
win_hamming = signal.hamming(1024)

# average the df_FFR
df_FFR_avg = pd.DataFrame()
df_FFR_avg_win = pd.DataFrame()
# average test1 and test2
for i in range(704):
    # combine next two rows later
    df_FFR_avg_t = pd.DataFrame(df_FFR.iloc[2*i: 2*i+2, 0:1024].mean(axis=0).values.reshape(1,1024)) # average those two rows
    # without window function
    df_FFR_avg_t = pd.DataFrame(df_FFR_avg_t.iloc[0,:].values.reshape(1,1024)) # without window function
    # implement the window function
    df_FFR_avg_t_window = pd.DataFrame((df_FFR_avg_t.iloc[0,:] * win_hamming).values.reshape(1,1024))
    df_FFR_label = pd.DataFrame(df_FFR.iloc[2*i, 1024:1031].values.reshape(1,7))
    df_FFR_avg = df_FFR_avg.append(pd.concat([df_FFR_avg_t, df_FFR_label], axis=1, ignore_index=True))
    df_FFR_avg_win = df_FFR_avg_win.append(pd.concat([df_FFR_avg_t_window, df_FFR_label], axis=1, ignore_index=True))
    
# set the title of columns
df_FFR_avg.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_FFR_avg = df_FFR_avg.sort_values(by=["Condition", "Subject"])
df_FFR_avg = df_FFR_avg.reset_index(drop=True)
df_FFR_avg_win.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_FFR_avg_win = df_FFR_avg_win.sort_values(by=["Condition", "Subject"])
df_FFR_avg_win = df_FFR_avg_win.reset_index(drop=True)



# average all the subjects , test and retest and keep one sound levels
# filter by 'a vowel and 85Db'
df_FFR_avg_sorted = df_FFR_avg.sort_values(by=["Sound Level", "Vowel","Condition", "Subject"])
df_FFR_avg_sorted = df_FFR_avg_sorted.reset_index(drop=True)
df_FFR_avg_win_sorted = df_FFR_avg_win.sort_values(by=["Sound Level", "Vowel","Condition", "Subject"])
df_FFR_avg_win_sorted = df_FFR_avg_win_sorted.reset_index(drop=True)

# filter55 65 75 sound levels and keep 85dB
# keep vowel condition and subject
df_FFR_avg_85 = pd.DataFrame(df_FFR_avg_sorted.iloc[528:, :])
df_FFR_avg_85 = df_FFR_avg_85.reset_index(drop=True)
df_FFR_avg_win_85 = pd.DataFrame(df_FFR_avg_win_sorted.iloc[528:, :])
df_FFR_avg_win_85 = df_FFR_avg_win_85.reset_index(drop=True)

# this part was replaced by upper part based on what I need to do
'''
# average all the subjects , test and retest, different sound levels
# filter by 'a vowel and 85Db'
df_FFR_avg_sorted = df_FFR_avg.sort_values(by=["Vowel","Condition", "Subject", "Sound Level"])
df_FFR_avg_sorted = df_FFR_avg_sorted.reset_index(drop=True)

# average sound levels and
# keep vowel condition and subject
df_FFR_avg_vcs = pd.DataFrame()
for i in range(176):
    # combine next two rows later
    df_FFR_avg_vcs_t = pd.DataFrame(df_FFR_avg_sorted.iloc[4*i: 4*i+4, 0:1024].mean(axis=0).values.reshape(1,1024)) # average those two rows
    df_FFR_avg_vcs_label = pd.DataFrame(df_FFR_avg_sorted.iloc[4*i, 1024:1031].values.reshape(1,7))
    df_FFR_avg_vcs = df_FFR_avg_vcs.append(pd.concat([df_FFR_avg_vcs_t, df_FFR_avg_vcs_label], axis=1, ignore_index=True), ignore_index=True)

# set the title of columns
df_FFR_avg_vcs.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
#df_FFR_avg_vcs = df_FFR_avg_vcs.sort_values(by=["Condition", "Subject"])
'''



'''
# filter by 'a vowel and 85Db'
df_FFR_a_85_test1 = df_FFR[(df_FFR['Vowel'] == 'a vowel') & (df_FFR['Sound Level'] == '85')]
df_FFR_a_85_test1 = df_FFR_a_85_test1.reset_index(drop=True)

df_FFR_a_85_avg = pd.DataFrame()
# average test1 and test2
for i in range(44):
    df_FFR_a_85_avg_t = pd.DataFrame(df_FFR_a_85_test1.iloc[2*i: 2*i+2, 0:1024].mean(axis=0).values.reshape(1,1024))
    df_FFR_a_85_label = pd.DataFrame(df_FFR_a_85_test1.iloc[2*i, 1024:1031].values.reshape(1,7))
    df_FFR_a_85_avg = df_FFR_a_85_avg.append(pd.concat([df_FFR_a_85_avg_t, df_FFR_a_85_label], axis=1, ignore_index=True))
# set the title of columns
df_FFR_a_85_avg.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_FFR_a_85_avg = df_FFR_a_85_avg.sort_values(by=["Condition", "Subject"])
df_FFR_a_85_avg = df_FFR_a_85_avg.reset_index(drop=True)
'''

     





 
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

n3 = 40960


# zero padding

# for df_FFR
df_FFR_data = df_FFR.iloc[:, :1024]
df_FFR_label = df_FFR.iloc[:, 1024:]
df_FFR_mid = pd.DataFrame(np.zeros((1408, 95036)))
df_FFR_withzero = pd.concat([df_FFR_data, df_FFR_mid, df_FFR_label], axis=1)
# rename columns
df_FFR_withzero.columns = np.append(np.arange(96060), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])

# for df_FFR_avg_85
df_FFR_avg_85_data = df_FFR_avg_85.iloc[:, :1024]
df_FFR_avg_85_label = df_FFR_avg_85.iloc[:, 1024:]
df_FFR_avg_85_mid = pd.DataFrame(np.zeros((176, 95036)))
df_FFR_avg_85_withzero = pd.concat([df_FFR_avg_85_data, df_FFR_avg_85_mid, df_FFR_avg_85_label], axis=1)
# rename columns
df_FFR_avg_85_withzero.columns = np.append(np.arange(96060), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])

# df_FFR_avg_win_85
df_FFR_avg_win_85_data = df_FFR_avg_win_85.iloc[:, :1024]
df_FFR_avg_win_85_label = df_FFR_avg_win_85.iloc[:, 1024:]
df_FFR_avg_win_85_mid = pd.DataFrame(np.zeros((176, 95036)))
df_FFR_avg_win_85_withzero = pd.concat([df_FFR_avg_win_85_data, df_FFR_avg_win_85_mid, df_FFR_avg_win_85_label], axis=1)
df_FFR_avg_win_85_withzero.columns = np.append(np.arange(96060), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])



# concatenate AENU
temp1 = pd.concat([df_FFR_avg_85.iloc[0:44, 0:1024].reset_index(drop=True),df_FFR_avg_85.iloc[44:88, 0:1024].reset_index(drop=True)], axis=1)
temp2 = pd.concat([df_FFR_avg_85.iloc[88:132, 0:1024].reset_index(drop=True), df_FFR_avg_85.iloc[132:176, 0:1024].reset_index(drop=True)], axis=1)
df_FFR_avg_85_aenu = pd.concat([temp1, temp2], axis=1, ignore_index=True)
 
df_FFR_avg_85_aenu_withzero = pd.concat([df_FFR_avg_85_aenu, pd.DataFrame(np.zeros((44, 36864)))] , axis=1)

'''
# test##############
# test(detrend)
temp_test = np.asarray(df_FFR_avg_85_data.iloc[0, 0:1024])
temp_test_detrend = signal.detrend(temp_test)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(temp_test)
plt.subplot(2, 1, 2)
plt.plot(temp_test_detrend)
plt.show()
# the raw data is already DC removed


# test(zero padding)
temp_FFR_1 = df_FFR_withzero.iloc[0, 0:1024]
temp_FFR_2= df_FFR_withzero.iloc[0, 0:9606]
temp_amplitude_spectrum_1 = np.abs((fftpack.fft(temp_FFR_1)/n)[range(int(n/2))])
temp_amplitude_spectrum_2 = np.abs((fftpack.fft(temp_FFR_2)/n2)[range(int(n2/2))])

plt.figure()
plt.subplot(2, 1, 1)
markers1 = [11, 21, 32, 43, 53, 64, 75]
# which corresponds to 100 200....700Hz in frequency domain
plt.plot(temp_amplitude_spectrum_1, '-D', markevery=markers1)
plt.xlim(0, 100)
plt.title('without zero padding')
plt.subplot(2, 1, 2)
#markers2 = [100, 200, 300, 400, 500, 600, 700]
markers2 = [99, 199, 299, 399, 499, 599, 599]
# which corresponds to 100 200....700Hz in frequency domain
plt.plot(temp_amplitude_spectrum_2, '-D', markevery=markers2)
plt.xlim(0, 1000)
# plt.xscale('linear')
plt.title('with zero padding')
plt.show()
# #################
'''


# Calculate the Amplitude Spectrum

# create a new dataframe with zero-padding amplitude spectrum

'''
# for df_FFR
df_as_7= pd.DataFrame()
for i in range(1408):
    temp_FFR = df_FFR_avg_85_withzero.iloc[i, 0:96060]
    temp_as = np.abs((fftpack.fft(temp_FFR)/n2)[range(int(n2/2))])
    #df_as_7 = pd.concat([df_as_7, temp_as_7_t], axis=0)
    df_as_7 = df_as_7.append(pd.DataFrame(np.array([temp_as[1000], temp_as[2000], temp_as[3000], temp_as[4000], \
                                                    temp_as[5000], temp_as[6000], temp_as[7000]]).reshape(1,7)), ignore_index = True)
   
df_as_7 = pd.concat([df_as_7, df_FFR_label], axis=1) # add labels on it

# filter by 'a vowel and 85Db'
df_as_7_test1 = df_as_7[(df_as_7['Vowel'] == 'a vowel') & (df_as_7['Sound Level'] == '85')]
df_as_7_test1 = df_as_7_test1.reset_index(drop=True)
'''





# for df_FFR_avg_vcs_withzero
df_as_85= pd.DataFrame()
df_as13_85= pd.DataFrame()
df_as_win_85= pd.DataFrame()
df_as13_win_85= pd.DataFrame()
for i in range(176):
    #temp_aenu_FFR = df_FFR_avg_aenu_withzero.iloc[i, 0:9606]
    temp_as = np.abs((fftpack.fft(df_FFR_avg_85_withzero.iloc[i, 0:96060])/n2)[range(int(n2/2))])
    df_as_85 = df_as_85.append(pd.DataFrame(temp_as.reshape(1,48030)), ignore_index = True)
    df_as13_85 = df_as13_85.append(pd.DataFrame(np.array([temp_as[1000], temp_as[2000], temp_as[3000], temp_as[4000], \
                                                            temp_as[5000], temp_as[6000], temp_as[7000], temp_as[8000], \
                                                            temp_as[9000], temp_as[10000], temp_as[11000], temp_as[12000], \
                                                            temp_as[13000]]).reshape(1,13)), ignore_index = True)
    temp_as_win = np.abs((fftpack.fft(df_FFR_avg_win_85_withzero.iloc[i, 0:96060])/n2)[range(int(n2/2))])
    df_as_win_85 = df_as_win_85.append(pd.DataFrame(temp_as_win.reshape(1,48030)), ignore_index = True)
    df_as13_win_85 = df_as13_win_85.append(pd.DataFrame(np.array([temp_as[1000], temp_as[2000], temp_as[3000], temp_as[4000], \
                                                            temp_as[5000], temp_as[6000], temp_as[7000], temp_as[8000], \
                                                            temp_as[9000], temp_as[10000], temp_as[11000], temp_as[12000], \
                                                            temp_as[13000]]).reshape(1,13)), ignore_index = True)

df_as_85 = pd.concat([df_as_85, df_FFR_avg_85_label], axis=1) # add labels on it
df_as13_85 = pd.concat([df_as13_85, df_FFR_avg_85_label], axis=1) # add labels on it
df_as_win_85 = pd.concat([df_as_win_85, df_FFR_avg_win_85_label], axis=1) # add labels on it
df_as13_win_85 = pd.concat([df_as13_win_85, df_FFR_avg_win_85_label], axis=1) # add labels on it


# for FFR_aenu
df_aenu_as_85= pd.DataFrame()
for i in range(44):
    #temp_aenu_FFR = df_FFR_avg_aenu_withzero.iloc[i, 0:9606]
    temp_as2 = np.abs((fftpack.fft(df_FFR_avg_85_aenu.iloc[i, 0:4096])/4096)[range(int(4096/2))])
    df_aenu_as_85 = df_aenu_as_85.append(pd.DataFrame(temp_as2.reshape(1,2048)), ignore_index = True)    
#df_aenu_as_85 = pd.concat([df_aenu_as_85, df_FFR_avg_85_label], axis=1) # add labels on it


# as_aenu
# firstly, convert signal into frequency domain, then concatenate it.
temp3 = pd.concat([df_as_85.iloc[0:44, 0:10000].reset_index(drop=True), df_as_85.iloc[44:88, 0:10000].reset_index(drop=True)], axis=1)
temp4 = pd.concat([df_as_85.iloc[88:132, 0:10000].reset_index(drop=True), df_as_85.iloc[132:176, 0:10000].reset_index(drop=True)], axis=1)
df_as_aenu_85 = pd.concat([temp3, temp4], axis=1, ignore_index=True)

temp5 = pd.concat([df_as_win_85.iloc[0:44, 0:10000].reset_index(drop=True), df_as_win_85.iloc[44:88, 0:10000].reset_index(drop=True)], axis=1)
temp6 = pd.concat([df_as_win_85.iloc[88:132, 0:10000].reset_index(drop=True), df_as_win_85.iloc[132:176, 0:10000].reset_index(drop=True)], axis=1)
df_as_win_aenu_85 = pd.concat([temp5, temp6], axis=1, ignore_index=True)


# as7_aenu 
temp7 = pd.concat([df_as13_85.iloc[0:44, 0:13].reset_index(drop=True), df_as13_85.iloc[44:88, 0:13].reset_index(drop=True)], axis=1)
temp8 = pd.concat([df_as13_85.iloc[88:132, 0:13].reset_index(drop=True), df_as13_85.iloc[132:176, 0:13].reset_index(drop=True)], axis=1)
df_as7_aenu_85 = pd.concat([temp7, temp8], axis=1, ignore_index=True)

temp9 = pd.concat([df_as13_win_85.iloc[0:44, 0:13].reset_index(drop=True), df_as13_win_85.iloc[44:88, 0:13].reset_index(drop=True)], axis=1)
temp10 = pd.concat([df_as13_win_85.iloc[88:132, 0:13].reset_index(drop=True), df_as13_win_85.iloc[132:176, 0:13].reset_index(drop=True)], axis=1)
df_as7_win_aenu_85 = pd.concat([temp9, temp10], axis=1, ignore_index=True)



'''
# average test1 and test2
df_as_7_avg = pd.DataFrame()

for i in range(44):
    df_as_7_avg1 = pd.DataFrame(df_as_7_test1.iloc[2*i: 2*i+1, 0:7].mean(axis=0).values.reshape(1,7))
    df_as_7_label = pd.DataFrame(df_as_7_test1.iloc[2*i, 7:14].values.reshape(1,7))
    df_as_7_avg_t = pd.concat([df_as_7_avg1, df_as_7_label], axis=1, ignore_index=True)
    df_as_7_avg = df_as_7_avg.append(df_as_7_avg_t)

# set the title of columns
df_as_7_avg.columns = np.append(np.arange(7), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_as_7_avg = df_as_7_avg.sort_values(by=["Condition", "Subject"])
df_as_7_avg = df_as_7_avg.reset_index(drop=True)
'''

'''
# set a normalized AS
df_as_7_avg_data= pd.DataFrame(df_as_7_avg.iloc[:, 0:7].astype(float))
df_as_7_avg_sum= pd.DataFrame(df_as_7_avg.iloc[:, 0:7]).sum(axis=1)
df_as_7_avg_label= pd.DataFrame(df_as_7_avg.iloc[:, 7:14])
# normalize
df_as_7_avg_norm = df_as_7_avg_data.div(df_as_7_avg_sum, axis=0)
# add label
df_as_7_avg_norm = pd.concat([df_as_7_avg_norm, df_as_7_avg_label], axis=1, ignore_index=True)
'''





# Calculate correlation
# FFR
'''
corr_FFR_avg_85_a = df_FFR_avg_85.iloc[0:44, 0:1024].T.corr(method='pearson').iloc[22:44, 0:22]
corr_FFR_avg_85_e = df_FFR_avg_85.iloc[44:88, 0:1024].T.corr(method='pearson').iloc[22:44, 0:22]
corr_FFR_avg_85_n = df_FFR_avg_85.iloc[88:132, 0:1024].T.corr(method='pearson').iloc[22:44, 0:22]
corr_FFR_avg_85_u = df_FFR_avg_85.iloc[132:176, 0:1024].T.corr(method='pearson').iloc[22:44, 0:22]
'''
# corr_FFR_avg_85_aenu = df_FFR_avg_85_aenu.iloc[:, 0:4096].T.corr(method='pearson').iloc[22:44, 0:22]

'''
corr_FFR_avg_85_a_t = df_FFR_avg_85.iloc[0:44, 0:1024].T.corr(method='pearson').iloc[0:22, 0:22]
corr_FFR_avg_85_e_t = df_FFR_avg_85.iloc[44:88, 0:1024].T.corr(method='pearson').iloc[0:22, 0:22]
corr_FFR_avg_85_n_t = df_FFR_avg_85.iloc[88:132, 0:1024].T.corr(method='pearson').iloc[0:22, 0:22]
corr_FFR_avg_85_u_t = df_FFR_avg_85.iloc[132:176, 0:1024].T.corr(method='pearson').iloc[0:22, 0:22]
corr_FFR_avg_85_a_re = df_FFR_avg_85.iloc[0:44, 0:1024].T.corr(method='pearson').iloc[22:44, 22:44]
corr_FFR_avg_85_e_re = df_FFR_avg_85.iloc[44:88, 0:1024].T.corr(method='pearson').iloc[22:44, 22:44]
corr_FFR_avg_85_n_re = df_FFR_avg_85.iloc[88:132, 0:1024].T.corr(method='pearson').iloc[22:44, 22:44]
corr_FFR_avg_85_u_re = df_FFR_avg_85.iloc[132:176, 0:1024].T.corr(method='pearson').iloc[22:44, 22:44]
'''

# AS
'''
corr_as_win_85_a = df_as_win_85.iloc[0:44, 0:12000].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as_win_85_e = df_as_win_85.iloc[44:88, 0:12000].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as_win_85_n = df_as_win_85.iloc[88:132, 0:12000].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as_win_85_u = df_as_win_85.iloc[132:176, 0:12000].T.corr(method='pearson').iloc[22:44, 0:22]
'''

corr_as_aenu_85 = df_as_aenu_85.iloc[0:44, 0:40000].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as_win_aenu_85 = df_as_win_aenu_85.iloc[0:44, 0:40000].T.corr(method='pearson').iloc[22:44, 0:22]


'''
corr_as_85_a = df_as_85.iloc[0:44, 0:48030].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as_85_e = df_as_85.iloc[44:88, 0:48030].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as_85_n = df_as_85.iloc[88:132, 0:48030].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as_85_u = df_as_85.iloc[132:176, 0:48030].T.corr(method='pearson').iloc[22:44, 0:22]


corr_as_85_a_t = df_as_85.iloc[0:44, 0:48030].T.corr(method='pearson').iloc[0:22, 0:22]
corr_as_85_e_t = df_as_85.iloc[44:88, 0:48030].T.corr(method='pearson').iloc[0:22, 0:22]
corr_as_85_n_t = df_as_85.iloc[88:132, 0:48030].T.corr(method='pearson').iloc[0:22, 0:22]
corr_as_85_u_t = df_as_85.iloc[132:176, 0:48030].T.corr(method='pearson').iloc[0:22, 0:22]
corr_as_85_a_re = df_as_85.iloc[0:44, 0:48030].T.corr(method='pearson').iloc[22:44, 22:44]
corr_as_85_e_re = df_as_85.iloc[44:88, 0:48030].T.corr(method='pearson').iloc[22:44, 22:44]
corr_as_85_n_re = df_as_85.iloc[88:132, 0:48030].T.corr(method='pearson').iloc[22:44, 22:44]
corr_as_85_u_re = df_as_85.iloc[132:176, 0:48030].T.corr(method='pearson').iloc[22:44, 22:44]
'''

#AS7
'''
corr_as7_85_a = df_as7_85.iloc[0:44, 0:7].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as7_85_e = df_as7_85.iloc[44:88, 0:7].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as7_85_n = df_as7_85.iloc[88:132, 0:7].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as7_85_u = df_as7_85.iloc[132:176, 0:7].T.corr(method='pearson').iloc[22:44, 0:22]

corr_as7_85_a_t = df_as7_85.iloc[0:44, 0:7].T.corr(method='pearson').iloc[0:22, 0:22]
corr_as7_85_e_t = df_as7_85.iloc[44:88, 0:7].T.corr(method='pearson').iloc[0:22, 0:22]
corr_as7_85_n_t = df_as7_85.iloc[88:132, 0:7].T.corr(method='pearson').iloc[0:22, 0:22]
corr_as7_85_u_t = df_as7_85.iloc[132:176, 0:7].T.corr(method='pearson').iloc[0:22, 0:22]
corr_as7_85_a_re = df_as7_85.iloc[0:44, 0:7].T.corr(method='pearson').iloc[22:44, 22:44]
corr_as7_85_e_re = df_as7_85.iloc[44:88, 0:7].T.corr(method='pearson').iloc[22:44, 22:44]
corr_as7_85_n_re = df_as7_85.iloc[88:132, 0:7].T.corr(method='pearson').iloc[22:44, 22:44]
corr_as7_85_u_re = df_as7_85.iloc[132:176, 0:7].T.corr(method='pearson').iloc[22:44, 22:44]
'''
corr_as7_aenu_85 = df_as7_aenu_85.iloc[0:44, 0:28].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as7_win_aenu_85 = df_as7_win_aenu_85.iloc[0:44, 0:28].T.corr(method='pearson').iloc[22:44, 0:22]



'''
# auto-correlation
corr_FFR_a_85_retest = df_FFR_a_85_avg.iloc[0:22, 0:7].T.corr(method='pearson')
corr_FFR_a_85_test = df_FFR_a_85_avg.iloc[22:44, 0:7].T.corr(method='pearson')
# cross-correlation
corr_FFR_a_85_r_t = df_FFR_a_85_avg.iloc[:, 0:7].T.corr(method='pearson')
# correlation matrix of test and retest
corr_FFR_a_85_r_t_part = corr_FFR_a_85_r_t.iloc[22:44, 0:22]


# auto-correlation
corr_as_retest = df_as_7_avg.iloc[0:22, 0:7].T.corr(method='pearson')
corr_as_test = df_as_7_avg.iloc[22:44, 0:7].T.corr(method='pearson')
# cross-correlation
corr_as_r_t = df_as_7_avg.iloc[:, 0:7].T.corr(method='pearson')

# Calculate correlation(normalized)
# auto-correlation
corr_as_norm_retest = df_as_7_avg_norm.iloc[0:22, 0:7].T.corr(method='pearson')
corr_as_norm_test = df_as_7_avg_norm.iloc[22:44, 0:7].T.corr(method='pearson')
# cross-correlation
corr_as_norm_r_t = df_as_7_avg_norm.iloc[:, 0:7].T.corr(method='pearson')
'''


# shrink
## shrink the correlation range from 0.3 to 1
# AS
corr_as_aenu_85_shrink_03_1 = shrink_value_03_1(corr_as_aenu_85)
corr_as_win_aenu_85_shrink_03_1 = shrink_value_03_1(corr_as_win_aenu_85)
# AS7
corr_as7_aenu_85_shrink_03_1 = shrink_value_03_1(corr_as7_aenu_85)
corr_as7_win_aenu_85_shrink_03_1 = shrink_value_03_1(corr_as7_win_aenu_85)


# shrink the correlation range from 0.5 to 1
# AS
#corr_as_aenu_85_shrink_05_1 = shrink_value_05_1(corr_as_aenu_85)
#corr_as_win_aenu_85_shrink_05_1 = shrink_value_05_1(corr_as_win_aenu_85)
# AS7
#corr_as7_aenu_85_shrink_05_1 = shrink_value_05_1(corr_as7_aenu_85)
#corr_as7_win_aenu_85_shrink_05_1 = shrink_value_05_1(corr_as7_win_aenu_85)




# plot the figure


'''
# Correlation Matrix

# FFR
correlation_matrix(corr_FFR_avg_85_a, 'cross correlation of 85dB a_vowel in time domain')
correlation_matrix(corr_FFR_avg_85_e, 'cross correlation of 85dB e_vowel in time domain')
correlation_matrix(corr_FFR_avg_85_n, 'cross correlation of 85dB n_vowel in time domain')
correlation_matrix(corr_FFR_avg_85_u, 'cross correlation of 85dB u_vowel in time domain')

# AS
correlation_matrix(corr_as_85_a, 'cross correlation of 85dB a_vowel in frequency domain')
correlation_matrix(corr_as_85_e, 'cross correlation of 85dB e_vowel in frequency domain')
correlation_matrix(corr_as_85_n, 'cross correlation of 85dB n_vowel in frequency domain')
correlation_matrix(corr_as_85_u, 'cross correlation of 85dB u_vowel in frequency domain')

# AS7
correlation_matrix(corr_as7_85_a, 'cross correlation of 85dB a_vowel in frequency domain 7')
correlation_matrix(corr_as7_85_e, 'cross correlation of 85dB e_vowel in frequency domain 7')
correlation_matrix(corr_as7_85_n, 'cross correlation of 85dB n_vowel in frequency domain 7')
correlation_matrix(corr_as7_85_u, 'cross correlation of 85dB u_vowel in frequency domain 7')




# Correlation Matrix witn 0 and 1

# FFR
correlation_matrix_01(corr_FFR_avg_85_a, 'cross correlation of 85dB a_vowel in time domain')
#correlation_matrix_tt_01(corr_FFR_avg_85_a_t, 'cross correlation of 85dB a_vowel in time domain')
#correlation_matrix_rr_01(corr_FFR_avg_85_a_re, 'cross correlation of 85dB a_vowel in time domain')

correlation_matrix_01(corr_FFR_avg_85_e, 'cross correlation of 85dB e_vowel in time domain')
#correlation_matrix_tt_01(corr_FFR_avg_85_e_t, 'cross correlation of 85dB e_vowel in time domain')
#correlation_matrix_rr_01(corr_FFR_avg_85_e_re, 'cross correlation of 85dB e_vowel in time domain')

correlation_matrix_01(corr_FFR_avg_85_n, 'cross correlation of 85dB n_vowel in time domain')
#correlation_matrix_tt_01(corr_FFR_avg_85_n_t, 'cross correlation of 85dB n_vowel in time domain')
#correlation_matrix_rr_01(corr_FFR_avg_85_n_re, 'cross correlation of 85dB n_vowel in time domain')

correlation_matrix_01(corr_FFR_avg_85_u, 'cross correlation of 85dB u_vowel in time domain')
#correlation_matrix_tt_01(corr_FFR_avg_85_u_t, 'cross correlation of 85dB u_vowel in time domain')
#correlation_matrix_rr_01(corr_FFR_avg_85_u_re, 'cross correlation of 85dB u_vowel in time domain')


# Amplitude Spectrum
correlation_matrix_01(corr_as_85_a, 'cross correlation of 85dB a_vowel in frequency domain')
#correlation_matrix_tt_01(corr_as_85_a_t, 'cross correlation of 85dB a_vowel in frequency domain')
#correlation_matrix_rr_01(corr_as_85_a_re, 'cross correlation of 85dB a_vowel in frequency domain')

correlation_matrix_01(corr_as_85_e, 'cross correlation of 85dB e_vowel in frequency domain')
#correlation_matrix_tt_01(corr_as_85_e_t, 'cross correlation of 85dB e_vowel in frequency domain')
#correlation_matrix_rr_01(corr_as_85_e_re, 'cross correlation of 85dB e_vowel in frequency domain')

correlation_matrix_01(corr_as_85_n, 'cross correlation of 85dB n_vowel in frequency domain')
#correlation_matrix_tt_01(corr_as_85_n_t, 'cross correlation of 85dB n_vowel in frequency domain')
#correlation_matrix_rr_01(corr_as_85_n_re, 'cross correlation of 85dB n_vowel in frequency domain')

correlation_matrix_01(corr_as_85_u, 'cross correlation of 85dB u_vowel in frequency domain')
#correlation_matrix_tt_01(corr_as_85_u_t, 'cross correlation of 85dB u_vowel in frequency domain')
#correlation_matrix_rr_01(corr_as_85_u_re, 'cross correlation of 85dB u_vowel in frequency domain')


# Amplitude Spectrum 7 points
correlation_matrix_01(corr_as7_85_a, 'cross correlation of 85dB a_vowel in frequency domain 7')
#correlation_matrix_tt_01(corr_as7_85_a_t, 'cross correlation of 85dB a_vowel in frequency domain 7')
#correlation_matrix_rr_01(corr_as7_85_a_re, 'cross correlation of 85dB a_vowel in frequency domain 7')

correlation_matrix_01(corr_as7_85_e, 'cross correlation of 85dB e_vowel in frequency domain 7')
#correlation_matrix_tt_01(corr_as7_85_e_t, 'cross correlation of 85dB e_vowel in frequency domain 7')
#correlation_matrix_rr_01(corr_as7_85_e_re, 'cross correlation of 85dB e_vowel in frequency domain 7')

correlation_matrix_01(corr_as7_85_n, 'cross correlation of 85dB n_vowel in frequency domain 7')
#correlation_matrix_tt_01(corr_as7_85_n_t, 'cross correlation of 85dB n_vowel in frequency domain 7')
#correlation_matrix_rr_01(corr_as7_85_n_re, 'cross correlation of 85dB n_vowel in frequency domain 7')

correlation_matrix_01(corr_as7_85_u, 'cross correlation of 85dB u_vowel in frequency domain 7')
#correlation_matrix_tt_01(corr_as7_85_u_t, 'cross correlation of 85dB u_vowel in frequency domain 7')
#correlation_matrix_rr_01(corr_as7_85_u_re, 'cross correlation of 85dB u_vowel in frequency domain 7')
'''


# Correlation Matrix_comb

# FFR
'''
correlation_matrix_comb(corr_FFR_avg_85_a, 'cross correlation of 85dB a_vowel in time domain')
correlation_matrix_comb(corr_FFR_avg_85_e, 'cross correlation of 85dB e_vowel in time domain')
correlation_matrix_comb(corr_FFR_avg_85_n, 'cross correlation of 85dB n_vowel in time domain')
correlation_matrix_comb(corr_FFR_avg_85_u, 'cross correlation of 85dB u_vowel in time domain')
'''
# correlation_matrix_comb(corr_FFR_avg_85_aenu, 'cross correlation of 85dB aenu in time domain')


# AS
'''
correlation_matrix_comb(corr_as_win_85_a, 'cross correlation of 85dB a_vowel in frequency domain')
correlation_matrix_comb(corr_as_win_85_e, 'cross correlation of 85dB e_vowel in frequency domain')
correlation_matrix_comb(corr_as_win_85_n, 'cross correlation of 85dB n_vowel in frequency domain')
correlation_matrix_comb(corr_as_win_85_u, 'cross correlation of 85dB u_vowel in frequency domain')
'''
correlation_matrix_comb(corr_as_aenu_85_shrink_03_1, 'cross correlation of shrinked(0.3, 1) 85dB as_aenu in frequency domain')
#correlation_matrix_comb(corr_as_aenu_85_shrink_05_1, 'cross correlation of shrinked(0.5, 1) 85dB as_aenu in frequency domain')
correlation_matrix_comb(corr_as_win_aenu_85_shrink_03_1, 'cross correlation of shrinked(0.3, 1) 85dB as_aenu in frequency domain(hamming)')
#correlation_matrix_comb(corr_as_win_aenu_85_shrink_05_1, 'cross correlation of shrinked(0.5, 1) 85dB as_aenu in frequency domain(hamming)')


# AS7
'''
correlation_matrix_comb(corr_as7_85_a, 'cross correlation of 85dB a_vowel in frequency domain 7')
correlation_matrix_comb(corr_as7_85_e, 'cross correlation of 85dB e_vowel in frequency domain 7')
correlation_matrix_comb(corr_as7_85_n, 'cross correlation of 85dB n_vowel in frequency domain 7')
correlation_matrix_comb(corr_as7_85_u, 'cross correlation of 85dB u_vowel in frequency domain 7')
'''
correlation_matrix_comb(corr_as7_aenu_85_shrink_03_1, 'cross correlation of shrinked(0.3, 1) 85dB as7_aenu in frequency domain')
#correlation_matrix_comb(corr_as7_aenu_85_shrink_05_1, 'cross correlation of shrinked(0.5, 1) 85dB as7_aenu in frequency domain')
correlation_matrix_comb(corr_as7_win_aenu_85_shrink_03_1, 'cross correlation of shrinked(0.3, 1) 85dB as7_aenu in frequency domain(hamming)')
#correlation_matrix_comb(corr_as7_win_aenu_85_shrink_05_1, 'cross correlation of shrinked(0.5, 1) 85dB as7_aenu in frequency domain(hamming)')

