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
    corr_out1 = corr_in1.copy()
    # here dataframe.copy() must be used, otherwise input can also be changed when changing output
    for i in range (22):
        for j in range(22):
            if corr_in1.iloc[i, j] < 0.3:
                corr_out1.iloc[i, j] = 0.3
    return corr_out1


def shrink_value_05_1(corr_in2):
    corr_out2 = corr_in2.copy()
    # here dataframe.copy() must be used, otherwise input can also be changed when changing output
    for i2 in range (22):
        for j2 in range(22):
            if corr_in2.iloc[i2, j2] < 0.5:
                corr_out2.iloc[i2, j2] = 0.5
    return corr_out2


# not used!!!!!!!!!!!!

# normalize the complex signal series
def normalize_complex_arr(a):
    a_oo = a - a.real.min() - 1j*a.imag.min() # origin offsetted
    return a_oo/np.abs(a_oo).max()



# import the pkl file
#pkl_file=pd.read_pickle('/Users/bruce/Documents/uOttawa/Project/audio_brainstem_response/Data_BruceSunMaster_Studies/study2/study2DataFrame.pkl')
df_EFR=pd.read_pickle('/home/bruce/Dropbox/Project/4.Code for Linux/df_EFR.pkl')

# remove DC offset
df_EFR_detrend = pd.DataFrame()
for i in range(1408):
    # combine next two rows later
    df_EFR_detrend_data = pd.DataFrame(signal.detrend(df_EFR.iloc[i: i+1, 0:1024], type='constant').reshape(1,1024))
    df_EFR_label = pd.DataFrame(df_EFR.iloc[i, 1024:1031].values.reshape(1,7))
    df_EFR_detrend = df_EFR_detrend.append(pd.concat([df_EFR_detrend_data, df_EFR_label], axis=1, ignore_index=True))
# set the title of columns
df_EFR_detrend.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_EFR_detrend = df_EFR_detrend.reset_index(drop=True)
df_EFR = df_EFR_detrend


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
    # without window function
    df_EFR_avg_t = pd.DataFrame(df_EFR_avg_t.iloc[0,:].values.reshape(1,1024)) # without window function
    # implement the window function
    df_EFR_avg_t_window = pd.DataFrame((df_EFR_avg_t.iloc[0,:] * win_hamming).values.reshape(1,1024))
    df_EFR_label = pd.DataFrame(df_EFR.iloc[2*i, 1024:1031].values.reshape(1,7))
    df_EFR_avg = df_EFR_avg.append(pd.concat([df_EFR_avg_t, df_EFR_label], axis=1, ignore_index=True))
    df_EFR_avg_win = df_EFR_avg_win.append(pd.concat([df_EFR_avg_t_window, df_EFR_label], axis=1, ignore_index=True))
    
# set the title of columns
df_EFR_avg.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_EFR_avg = df_EFR_avg.sort_values(by=["Condition", "Subject"])
df_EFR_avg = df_EFR_avg.reset_index(drop=True)
df_EFR_avg_win.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_EFR_avg_win = df_EFR_avg_win.sort_values(by=["Condition", "Subject"])
df_EFR_avg_win = df_EFR_avg_win.reset_index(drop=True)



# average all the subjects , test and retest and keep one sound levels
# filter by 'a vowel and 85Db'
df_EFR_avg_sorted = df_EFR_avg.sort_values(by=["Sound Level", "Vowel","Condition", "Subject"])
df_EFR_avg_sorted = df_EFR_avg_sorted.reset_index(drop=True)
df_EFR_avg_win_sorted = df_EFR_avg_win.sort_values(by=["Sound Level", "Vowel","Condition", "Subject"])
df_EFR_avg_win_sorted = df_EFR_avg_win_sorted.reset_index(drop=True)

# filter55 65 75 sound levels and keep 85dB
# keep vowel condition and subject
df_EFR_avg_85 = pd.DataFrame(df_EFR_avg_sorted.iloc[528:, :])
df_EFR_avg_85 = df_EFR_avg_85.reset_index(drop=True)
df_EFR_avg_win_85 = pd.DataFrame(df_EFR_avg_win_sorted.iloc[528:, :])
df_EFR_avg_win_85 = df_EFR_avg_win_85.reset_index(drop=True)

# this part was replaced by upper part based on what I need to do
'''
# average all the subjects , test and retest, different sound levels
# filter by 'a vowel and 85Db'
df_EFR_avg_sorted = df_EFR_avg.sort_values(by=["Vowel","Condition", "Subject", "Sound Level"])
df_EFR_avg_sorted = df_EFR_avg_sorted.reset_index(drop=True)

# average sound levels and
# keep vowel condition and subject
df_EFR_avg_vcs = pd.DataFrame()
for i in range(176):
    # combine next two rows later
    df_EFR_avg_vcs_t = pd.DataFrame(df_EFR_avg_sorted.iloc[4*i: 4*i+4, 0:1024].mean(axis=0).values.reshape(1,1024)) # average those two rows
    df_EFR_avg_vcs_label = pd.DataFrame(df_EFR_avg_sorted.iloc[4*i, 1024:1031].values.reshape(1,7))
    df_EFR_avg_vcs = df_EFR_avg_vcs.append(pd.concat([df_EFR_avg_vcs_t, df_EFR_avg_vcs_label], axis=1, ignore_index=True), ignore_index=True)

# set the title of columns
df_EFR_avg_vcs.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
#df_EFR_avg_vcs = df_EFR_avg_vcs.sort_values(by=["Condition", "Subject"])
'''



'''
# filter by 'a vowel and 85Db'
df_EFR_a_85_test1 = df_EFR[(df_EFR['Vowel'] == 'a vowel') & (df_EFR['Sound Level'] == '85')]
df_EFR_a_85_test1 = df_EFR_a_85_test1.reset_index(drop=True)

df_EFR_a_85_avg = pd.DataFrame()
# average test1 and test2
for i in range(44):
    df_EFR_a_85_avg_t = pd.DataFrame(df_EFR_a_85_test1.iloc[2*i: 2*i+2, 0:1024].mean(axis=0).values.reshape(1,1024))
    df_EFR_a_85_label = pd.DataFrame(df_EFR_a_85_test1.iloc[2*i, 1024:1031].values.reshape(1,7))
    df_EFR_a_85_avg = df_EFR_a_85_avg.append(pd.concat([df_EFR_a_85_avg_t, df_EFR_a_85_label], axis=1, ignore_index=True))
# set the title of columns
df_EFR_a_85_avg.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_EFR_a_85_avg = df_EFR_a_85_avg.sort_values(by=["Condition", "Subject"])
df_EFR_a_85_avg = df_EFR_a_85_avg.reset_index(drop=True)
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

# for df_EFR
df_EFR_data = df_EFR.iloc[:, :1024]
df_EFR_label = df_EFR.iloc[:, 1024:]
df_EFR_mid = pd.DataFrame(np.zeros((1408, 95036)))
df_EFR_withzero = pd.concat([df_EFR_data, df_EFR_mid, df_EFR_label], axis=1)
# rename columns
df_EFR_withzero.columns = np.append(np.arange(96060), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])

# for df_EFR_avg_85
df_EFR_avg_85_data = df_EFR_avg_85.iloc[:, :1024]
df_EFR_avg_85_label = df_EFR_avg_85.iloc[:, 1024:]
df_EFR_avg_85_mid = pd.DataFrame(np.zeros((176, 95036)))
df_EFR_avg_85_withzero = pd.concat([df_EFR_avg_85_data, df_EFR_avg_85_mid, df_EFR_avg_85_label], axis=1)
# rename columns
df_EFR_avg_85_withzero.columns = np.append(np.arange(96060), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])

# df_EFR_avg_win_85
df_EFR_avg_win_85_data = df_EFR_avg_win_85.iloc[:, :1024]
df_EFR_avg_win_85_label = df_EFR_avg_win_85.iloc[:, 1024:]
df_EFR_avg_win_85_mid = pd.DataFrame(np.zeros((176, 95036)))
df_EFR_avg_win_85_withzero = pd.concat([df_EFR_avg_win_85_data, df_EFR_avg_win_85_mid, df_EFR_avg_win_85_label], axis=1)
df_EFR_avg_win_85_withzero.columns = np.append(np.arange(96060), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])



# concatenate AENU
temp1 = pd.concat([df_EFR_avg_85.iloc[0:44, 0:1024].reset_index(drop=True),df_EFR_avg_85.iloc[44:88, 0:1024].reset_index(drop=True)], axis=1)
temp2 = pd.concat([df_EFR_avg_85.iloc[88:132, 0:1024].reset_index(drop=True), df_EFR_avg_85.iloc[132:176, 0:1024].reset_index(drop=True)], axis=1)
df_EFR_avg_85_aenu = pd.concat([temp1, temp2], axis=1, ignore_index=True)
 
df_EFR_avg_85_aenu_withzero = pd.concat([df_EFR_avg_85_aenu, pd.DataFrame(np.zeros((44, 36864)))] , axis=1)

'''
# test##############
# test(detrend)
temp_test = np.asarray(df_EFR_avg_85_data.iloc[0, 0:1024])
temp_test_detrend = signal.detrend(temp_test)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(temp_test)
plt.subplot(2, 1, 2)
plt.plot(temp_test_detrend)
plt.show()
# the raw data is already DC removed


# test(zero padding)
temp_EFR_1 = df_EFR_withzero.iloc[0, 0:1024]
temp_EFR_2= df_EFR_withzero.iloc[0, 0:9606]
temp_amplitude_spectrum_1 = np.abs((fftpack.fft(temp_EFR_1)/n)[range(int(n/2))])
temp_amplitude_spectrum_2 = np.abs((fftpack.fft(temp_EFR_2)/n2)[range(int(n2/2))])

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
# for df_EFR
df_as_7= pd.DataFrame()
for i in range(1408):
    temp_EFR = df_EFR_avg_85_withzero.iloc[i, 0:96060]
    temp_as = np.abs((fftpack.fft(temp_EFR)/n2)[range(int(n2/2))])
    #df_as_7 = pd.concat([df_as_7, temp_as_7_t], axis=0)
    df_as_7 = df_as_7.append(pd.DataFrame(np.array([temp_as[1000], temp_as[2000], temp_as[3000], temp_as[4000], \
                                                    temp_as[5000], temp_as[6000], temp_as[7000]]).reshape(1,7)), ignore_index = True)

df_as_7 = pd.concat([df_as_7, df_EFR_label], axis=1) # add labels on it

# filter by 'a vowel and 85Db'
df_as_7_test1 = df_as_7[(df_as_7['Vowel'] == 'a vowel') & (df_as_7['Sound Level'] == '85')]
df_as_7_test1 = df_as_7_test1.reset_index(drop=True)
'''





# for df_EFR_avg_vcs_withzero
df_as_85= pd.DataFrame()
df_as7_85= pd.DataFrame()
df_as_win_85= pd.DataFrame()
df_as7_win_85= pd.DataFrame()
for i in range(176):
    #temp_aenu_EFR = df_EFR_avg_aenu_withzero.iloc[i, 0:9606]
    temp_as = np.abs((fftpack.fft(df_EFR_avg_85_withzero.iloc[i, 0:96060])/n2)[range(int(n2/2))])
    df_as_85 = df_as_85.append(pd.DataFrame(temp_as.reshape(1,48030)), ignore_index = True)
    df_as7_85 = df_as7_85.append(pd.DataFrame(np.array([temp_as[1000], temp_as[2000], temp_as[3000], temp_as[4000], \
                                                            temp_as[5000], temp_as[6000], temp_as[7000]]).reshape(1,7)), ignore_index = True)
    temp_as_win = np.abs((fftpack.fft(df_EFR_avg_win_85_withzero.iloc[i, 0:96060])/n2)[range(int(n2/2))])
    df_as_win_85 = df_as_win_85.append(pd.DataFrame(temp_as_win.reshape(1,48030)), ignore_index = True)
    df_as7_win_85 = df_as7_win_85.append(pd.DataFrame(np.array([temp_as_win[1000], temp_as_win[2000], temp_as_win[3000], temp_as_win[4000], \
                                                            temp_as_win[5000], temp_as_win[6000], temp_as_win[7000]]).reshape(1,7)), ignore_index = True)

df_as_85 = pd.concat([df_as_85, df_EFR_avg_85_label], axis=1) # add labels on it
df_as7_85 = pd.concat([df_as7_85, df_EFR_avg_85_label], axis=1) # add labels on it
df_as_win_85 = pd.concat([df_as_win_85, df_EFR_avg_win_85_label], axis=1) # add labels on it
df_as7_win_85 = pd.concat([df_as7_win_85, df_EFR_avg_win_85_label], axis=1) # add labels on it


# for efr_aenu
df_aenu_as_85= pd.DataFrame()
for i in range(44):
    #temp_aenu_EFR = df_EFR_avg_aenu_withzero.iloc[i, 0:9606]
    temp_as2 = np.abs((fftpack.fft(df_EFR_avg_85_aenu.iloc[i, 0:4096])/4096)[range(int(4096/2))])
    df_aenu_as_85 = df_aenu_as_85.append(pd.DataFrame(temp_as2.reshape(1,2048)), ignore_index = True)
#df_aenu_as_85 = pd.concat([df_aenu_as_85, df_EFR_avg_85_label], axis=1) # add labels on it



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
# EFR

corr_EFR_avg_85_a = df_EFR_avg_85.iloc[0:44, 0:1024].T.corr(method='pearson').iloc[22:44, 0:22]
corr_EFR_avg_85_e = df_EFR_avg_85.iloc[44:88, 0:1024].T.corr(method='pearson').iloc[22:44, 0:22]
corr_EFR_avg_85_n = df_EFR_avg_85.iloc[88:132, 0:1024].T.corr(method='pearson').iloc[22:44, 0:22]
corr_EFR_avg_85_u = df_EFR_avg_85.iloc[132:176, 0:1024].T.corr(method='pearson').iloc[22:44, 0:22]

corr_EFR_avg_85_aenu = df_EFR_avg_85_aenu.iloc[:, 0:4096].T.corr(method='pearson').iloc[22:44, 0:22]
'''
corr_EFR_avg_85_a_t = df_EFR_avg_85.iloc[0:44, 0:1024].T.corr(method='pearson').iloc[0:22, 0:22]
corr_EFR_avg_85_e_t = df_EFR_avg_85.iloc[44:88, 0:1024].T.corr(method='pearson').iloc[0:22, 0:22]
corr_EFR_avg_85_n_t = df_EFR_avg_85.iloc[88:132, 0:1024].T.corr(method='pearson').iloc[0:22, 0:22]
corr_EFR_avg_85_u_t = df_EFR_avg_85.iloc[132:176, 0:1024].T.corr(method='pearson').iloc[0:22, 0:22]
corr_EFR_avg_85_a_re = df_EFR_avg_85.iloc[0:44, 0:1024].T.corr(method='pearson').iloc[22:44, 22:44]
corr_EFR_avg_85_e_re = df_EFR_avg_85.iloc[44:88, 0:1024].T.corr(method='pearson').iloc[22:44, 22:44]
corr_EFR_avg_85_n_re = df_EFR_avg_85.iloc[88:132, 0:1024].T.corr(method='pearson').iloc[22:44, 22:44]
corr_EFR_avg_85_u_re = df_EFR_avg_85.iloc[132:176, 0:1024].T.corr(method='pearson').iloc[22:44, 22:44]
'''


# AS


corr_as_win_85_a = df_as_win_85.iloc[0:44, 0:12000].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as_win_85_e = df_as_win_85.iloc[44:88, 0:12000].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as_win_85_n = df_as_win_85.iloc[88:132, 0:12000].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as_win_85_u = df_as_win_85.iloc[132:176, 0:12000].T.corr(method='pearson').iloc[22:44, 0:22]

corr_as_85_aenu = df_aenu_as_85.iloc[0:44, 0:2048].T.corr(method='pearson').iloc[22:44, 0:22]

# EFR + AS
df_EFR_avg_85_aenu_norm = df_EFR_avg_85_aenu.div((df_EFR_avg_85_aenu.max(axis=1) - df_EFR_avg_85_aenu.min(axis=1)), axis=0)
df_aenu_as_85_norm = df_aenu_as_85.div((df_aenu_as_85.max(axis=1) - df_aenu_as_85.min(axis=1)), axis=0)

df_aenu_sum_85 = pd.concat([df_EFR_avg_85_aenu_norm, df_aenu_as_85_norm.iloc[:, 0:535]], axis=1)
#corr_sum_85_aenu = df_aenu_sum_85.iloc[0:44, 0:2048].T.corr(method='pearson').iloc[22:44, 0:22]

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
corr_as7_85_a = df_as7_85.iloc[0:44, 0:7].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as7_85_e = df_as7_85.iloc[44:88, 0:7].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as7_85_n = df_as7_85.iloc[88:132, 0:7].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as7_85_u = df_as7_85.iloc[132:176, 0:7].T.corr(method='pearson').iloc[22:44, 0:22]

'''
corr_as7_85_a_t = df_as7_85.iloc[0:44, 0:7].T.corr(method='pearson').iloc[0:22, 0:22]
corr_as7_85_e_t = df_as7_85.iloc[44:88, 0:7].T.corr(method='pearson').iloc[0:22, 0:22]
corr_as7_85_n_t = df_as7_85.iloc[88:132, 0:7].T.corr(method='pearson').iloc[0:22, 0:22]
corr_as7_85_u_t = df_as7_85.iloc[132:176, 0:7].T.corr(method='pearson').iloc[0:22, 0:22]
corr_as7_85_a_re = df_as7_85.iloc[0:44, 0:7].T.corr(method='pearson').iloc[22:44, 22:44]
corr_as7_85_e_re = df_as7_85.iloc[44:88, 0:7].T.corr(method='pearson').iloc[22:44, 22:44]
corr_as7_85_n_re = df_as7_85.iloc[88:132, 0:7].T.corr(method='pearson').iloc[22:44, 22:44]
corr_as7_85_u_re = df_as7_85.iloc[132:176, 0:7].T.corr(method='pearson').iloc[22:44, 22:44]
'''


'''
# auto-correlation
corr_EFR_a_85_retest = df_EFR_a_85_avg.iloc[0:22, 0:7].T.corr(method='pearson')
corr_EFR_a_85_test = df_EFR_a_85_avg.iloc[22:44, 0:7].T.corr(method='pearson')
# cross-correlation
corr_EFR_a_85_r_t = df_EFR_a_85_avg.iloc[:, 0:7].T.corr(method='pearson')
# correlation matrix of test and retest
corr_EFR_a_85_r_t_part = corr_EFR_a_85_r_t.iloc[22:44, 0:22]


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
# shrink the correlation range from 0.3 to 1
# EFR

'''
corr_EFR_avg_85_a_shrink_03_1 = shrink_value_03_1(corr_EFR_avg_85_a)
corr_EFR_avg_85_e_shrink_03_1 = shrink_value_03_1(corr_EFR_avg_85_e)
corr_EFR_avg_85_n_shrink_03_1 = shrink_value_03_1(corr_EFR_avg_85_n)
corr_EFR_avg_85_u_shrink_03_1 = shrink_value_03_1(corr_EFR_avg_85_u)
'''
corr_EFR_avg_85_aenu_shrink_03_1 = shrink_value_03_1(corr_EFR_avg_85_aenu)

# AS

'''
corr_as_win_85_a_shrink_03_1 = shrink_value_03_1(corr_as_win_85_a)
corr_as_win_85_e_shrink_03_1 = shrink_value_03_1(corr_as_win_85_e)
corr_as_win_85_n_shrink_03_1 = shrink_value_03_1(corr_as_win_85_n)
corr_as_win_85_u_shrink_03_1 = shrink_value_03_1(corr_as_win_85_u)
'''

corr_as_85_aenu_shrink_03_1 = shrink_value_03_1(corr_as_85_aenu)


# shrink the correlation range from 0.5 to 1
# EFR
corr_EFR_avg_85_aenu_shrink_05_1 = shrink_value_05_1(corr_EFR_avg_85_aenu)
# AS
corr_as_85_aenu_shrink_05_1 = shrink_value_05_1(corr_as_85_aenu)


# test

# sum of time and frequency corelation matrix
corr_sum_avg_85_aenu = (corr_EFR_avg_85_aenu + corr_as_85_aenu).copy()

# max of time and frequency corelation matrix
# corr_max_avg_85_aenu = (corr_EFR_avg_85_aenu ? corr_as_85_aenu).copy()


# plot the figure
# in time and freq domain
df_EFR_avg_85_a = df_EFR_avg_85.iloc[0:44, 0:1024]
df_EFR_avg_85_e = df_EFR_avg_85.iloc[44:88, 0:1024]
df_EFR_avg_85_n = df_EFR_avg_85.iloc[88:132, 0:1024]
df_EFR_avg_85_u = df_EFR_avg_85.iloc[132:176, 0:1024]
r2_EFR_avg_85_aenu = pd.concat([df_EFR_avg_85_a.iloc[1, :], df_EFR_avg_85_e.iloc[1, :], 
                                df_EFR_avg_85_n.iloc[1, :], df_EFR_avg_85_u.iloc[1, :]], axis=0)
t2_EFR_avg_85_aenu = pd.concat([df_EFR_avg_85_a.iloc[23, :], df_EFR_avg_85_e.iloc[23, :], 
                                df_EFR_avg_85_n.iloc[23, :], df_EFR_avg_85_u.iloc[23, :]], axis=0)

r9_EFR_avg_85_aenu = pd.concat([df_EFR_avg_85_a.iloc[7, :], df_EFR_avg_85_e.iloc[7, :], 
                                df_EFR_avg_85_n.iloc[7, :], df_EFR_avg_85_u.iloc[7, :]], axis=0)
t9_EFR_avg_85_aenu = pd.concat([df_EFR_avg_85_a.iloc[29, :], df_EFR_avg_85_e.iloc[29, :], 
                                df_EFR_avg_85_n.iloc[29, :], df_EFR_avg_85_u.iloc[29, :]], axis=0)

# frequency domain
df_as_85_a = df_as_85.iloc[0:44, 0:48030]
df_as_85_e = df_as_85.iloc[44:88, 0:48030]
df_as_85_n = df_as_85.iloc[88:132, 0:48030]
df_as_85_u = df_as_85.iloc[132:176, 0:48030]

r1_as_85_aenu = df_aenu_as_85_norm.iloc[0,:]
r2_as_85_aenu = df_aenu_as_85_norm.iloc[1,:]
t2_as_85_aenu = df_aenu_as_85_norm.iloc[23, :]
r4_as_85_aenu = df_aenu_as_85_norm.iloc[3,:]
r7_as_85_aenu = df_aenu_as_85_norm.iloc[5,:]
r9_as_85_aenu = df_aenu_as_85_norm.iloc[7,:]
t9_as_85_aenu = df_aenu_as_85_norm.iloc[29, :]
r12_as_85_aenu = df_aenu_as_85_norm.iloc[9,:]
t12_as_85_aenu = df_aenu_as_85_norm.iloc[31, :]
r13_as_85_aenu = df_aenu_as_85_norm.iloc[10, :]
r14_as_85_aenu = df_aenu_as_85_norm.iloc[11,:]
t14_as_85_aenu = df_aenu_as_85_norm.iloc[33, :]
r15_as_85_aenu = df_aenu_as_85_norm.iloc[12,:]
t15_as_85_aenu = df_aenu_as_85_norm.iloc[34, :]
r16_as_85_aenu = df_aenu_as_85_norm.iloc[13, :]
r20_as_85_aenu = df_aenu_as_85_norm.iloc[17,:]
t20_as_85_aenu = df_aenu_as_85_norm.iloc[39, :]
r21_as_85_aenu = df_aenu_as_85_norm.iloc[18,:]
t21_as_85_aenu = df_aenu_as_85_norm.iloc[40, :]

# subject 2

# EFR
# subject2 -> 1 & 23

subject_num1 = 1
subject_num2 = 23
'''
x1_label = np.arange(0, 1024, 1)
x1_label_concat = np.arange(0, 4096, 1)
fig1, axs1 = plt.subplots(3, 2)
axs1[0,0].plot(x1_label, df_EFR_avg_85_a.iloc[subject_num1, :], label='retest')
axs1[0,0].plot(x1_label, df_EFR_avg_85_a.iloc[subject_num2, :], label='test')
axs1[0,0].legend(loc='upper right')
axs1[0,0].set_title('s2_EFR_avg_85_a')
axs1[0,1].plot(x1_label, df_EFR_avg_85_e.iloc[subject_num1, :], x1_label, df_EFR_avg_85_e.iloc[subject_num2, :])
axs1[0,1].set_title('s2_EFR_avg_85_e')
axs1[1,0].plot(x1_label, df_EFR_avg_85_n.iloc[subject_num1, :], x1_label, df_EFR_avg_85_n.iloc[subject_num2, :])
axs1[1,0].set_title('s2_EFR_avg_85_n')
axs1[1,1].plot(x1_label, df_EFR_avg_85_u.iloc[subject_num1, :], x1_label, df_EFR_avg_85_u.iloc[subject_num2, :])
axs1[1,1].set_title('s2_EFR_avg_85_u')
axs1[2,0].plot(x1_label_concat, r2_EFR_avg_85_aenu, x1_label_concat, t2_EFR_avg_85_aenu)
axs1[2,0].set_title('s2_EFR_avg_85_aenu')
plt.show()
'''

#AS
x2_label = np.arange(0, 48030, 1)
x2_label_concat = np.arange(0,2048, 1)
fig2, axs2 = plt.subplots(3, 2)
axs2[0,0].plot(x2_label, df_as_85_a.iloc[subject_num1, :], label='retest')
axs2[0,0].plot(x2_label, df_as_85_a.iloc[subject_num2, :], label='test')
axs2[0,0].set_xlim(0,13000) # 0 to 1300Hz
axs2[0,0].legend(loc='upper right')
axs2[0,0].set_title('s2_as_85_a')
axs2[0,1].plot(x2_label, df_as_85_e.iloc[subject_num1, :], x2_label, df_as_85_e.iloc[subject_num2, :])
axs2[0,1].set_xlim(0,13000) # 0 to 1300Hz
axs2[0,1].set_title('s2_as_85_e')
axs2[1,0].plot(x2_label, df_as_85_n.iloc[subject_num1, :], x2_label, df_as_85_n.iloc[subject_num2, :])
axs2[1,0].set_xlim(0,13000) # 0 to 1300Hz
axs2[1,0].set_title('s2_as_85_n')
axs2[1,1].plot(x2_label, df_as_85_u.iloc[subject_num1, :], x2_label, df_as_85_u.iloc[subject_num2, :])
axs2[1,1].set_xlim(0,13000) # 0 to 1300Hz
axs2[1,1].set_title('s2_as_85_u')
axs2[2,0].plot(x2_label_concat, r2_as_85_aenu, x2_label_concat, t2_as_85_aenu)
axs2[2,0].set_title('s2_as_85_aenu')
axs2[2,0].set_xlim(0,535) # 0 to 1300 Hz
axs2[2,1].plot(x2_label_concat, t2_as_85_aenu, label='test2')
axs2[2,1].plot(x2_label_concat, r16_as_85_aenu, label='retest16')
axs2[2,1].set_title('t2_as_85_aenu vs r16_as_85_aenu (rank#1 in freq domain)')
axs2[2,1].set_xlim(0,535) # 0 to 1300 Hz
plt.show()


# subject 9
subject_num1 = 7
subject_num2 = 29
# EFR
'''
x1_label = np.arange(0, 1024, 1)
x1_label_concat = np.arange(0, 4096, 1)
fig3, axs3 = plt.subplots(3, 2)
axs3[0,0].plot(x1_label, df_EFR_avg_85_a.iloc[subject_num1, :], label='retest')
axs3[0,0].plot(x1_label, df_EFR_avg_85_a.iloc[subject_num2, :], label='test')
axs3[0,0].legend(loc='upper right')
axs3[0,0].set_title('s9_EFR_avg_85_a')
axs3[0,1].plot(x1_label, df_EFR_avg_85_e.iloc[subject_num1, :], x1_label, df_EFR_avg_85_e.iloc[subject_num2, :])
axs3[0,1].set_title('s9_EFR_avg_85_e')
axs3[1,0].plot(x1_label, df_EFR_avg_85_n.iloc[subject_num1, :], x1_label, df_EFR_avg_85_n.iloc[subject_num2, :])
axs3[1,0].set_title('s9_EFR_avg_85_n')
axs3[1,1].plot(x1_label, df_EFR_avg_85_u.iloc[subject_num1, :], x1_label, df_EFR_avg_85_u.iloc[subject_num2, :])
axs3[1,1].set_title('s9_EFR_avg_85_u')
axs3[2,0].plot(x1_label_concat, r9_EFR_avg_85_aenu, label='retest9')
axs3[2,0].plot(x1_label_concat, t9_EFR_avg_85_aenu, label='test9')
axs3[2,0].legend(loc='upper right')
axs3[2,0].set_title('s9_EFR_avg_85_aenu')
plt.show()
'''
#AS
x2_label = np.arange(0, 48030, 1)
x2_label_concat = np.arange(0,2048, 1)
fig4, axs4 = plt.subplots(3, 2)
axs4[0,0].plot(x2_label, df_as_85_a.iloc[subject_num1, :], label='retest')
axs4[0,0].plot(x2_label, df_as_85_a.iloc[subject_num2, :], label='test')
axs4[0,0].set_xlim(0,13000) # 0 to 1300Hz
axs4[0,0].legend(loc='upper right')
axs4[0,0].set_title('s9_as_85_a')
axs4[0,1].plot(x2_label, df_as_85_e.iloc[subject_num1, :], x2_label, df_as_85_e.iloc[subject_num2, :])
axs4[0,1].set_xlim(0,13000) # 0 to 1300Hz
axs4[0,1].set_title('s9_as_85_e')
axs4[1,0].plot(x2_label, df_as_85_n.iloc[subject_num1, :], x2_label, df_as_85_n.iloc[subject_num2, :])
axs4[1,0].set_xlim(0,13000) # 0 to 1300Hz
axs4[1,0].set_title('s9_as_85_n')
axs4[1,1].plot(x2_label, df_as_85_u.iloc[subject_num1, :], x2_label, df_as_85_u.iloc[subject_num2, :])
axs4[1,1].set_xlim(0,13000) # 0 to 1300Hz
axs4[1,1].set_title('s9_as_85_u')
axs4[2,0].plot(x2_label_concat, r9_as_85_aenu, label='retest9')
axs4[2,0].plot(x2_label_concat, t9_as_85_aenu, label='test9')
axs4[2,0].legend(loc='upper right')
axs4[2,0].set_title('s9_as_85_aenu')
axs4[2,0].set_xlim(0,535) # 0 to 1300 Hz
plt.show()




# correlation matrix for freq domain (not #1)
# Subject 2 9 12
x2_label = np.arange(0, 48030, 1)
x2_label_concat = np.arange(0,2048, 1)
fig5, axs5 = plt.subplots(3, 2)
#T2R2 # T2R16
axs5[0,0].plot(x2_label_concat, r2_as_85_aenu, label='retest2')
axs5[0,0].plot(x2_label_concat, t2_as_85_aenu, label='test2')
axs5[0,0].legend(loc='upper right')
axs5[0,0].set_title('s2_as_85_aenu (0.866496, rank#3)')
axs5[0,0].set_xlim(0,535) # 0 to 1300 Hz

axs5[0,1].plot(x2_label_concat, r16_as_85_aenu, label='retest16')
axs5[0,1].plot(x2_label_concat, t2_as_85_aenu, label='test2')
axs5[0,1].legend(loc='upper right')
axs5[0,1].set_title('r16_as_85_aenu vs t2_as_85_aenu (0.871411, rank#1)')
axs5[0,1].set_xlim(0,535) # 0 to 1300 Hz

# R9T9 R13T9
axs5[1,0].plot(x2_label_concat, r9_as_85_aenu, label='retest9')
axs5[1,0].plot(x2_label_concat, t9_as_85_aenu, label='test9')
axs5[1,0].legend(loc='upper right')
axs5[1,0].set_title('s9_as_85_aenu (0.824085, rank#4)')
axs5[1,0].set_xlim(0,535) # 0 to 1300 Hz

axs5[1,1].plot(x2_label_concat, r13_as_85_aenu, label='retest13')
axs5[1,1].plot(x2_label_concat, t9_as_85_aenu, label='test9')
axs5[1,1].legend(loc='upper right')
axs5[1,1].set_title('r13_as_85_aenu vs. t9_as_85_aenu (0.838825, rank#1)')
axs5[1,1].set_xlim(0,535) # 0 to 1300 Hz

# T12R12 T12R4
axs5[2,0].plot(x2_label_concat, r12_as_85_aenu, label='retest12')
axs5[2,0].plot(x2_label_concat, t12_as_85_aenu, label='test12')
axs5[2,0].legend(loc='upper right')
axs5[2,0].set_title('s12_as_85_aenu (0.868822, rank#4)')
axs5[2,0].set_xlim(0,535) # 0 to 1300 Hz

axs5[2,1].plot(x2_label_concat, r4_as_85_aenu, label='retest4')
axs5[2,1].plot(x2_label_concat, t12_as_85_aenu, label='test12')
axs5[2,1].legend(loc='upper right')
axs5[2,1].set_title('r4_as_85_aenu vs. t12_as_85_aenu (0.892162, rank#1)')
axs5[2,1].set_xlim(0,535) # 0 to 1300 Hz
plt.show()


# Subject 14 15


x2_label = np.arange(0, 48030, 1)
x2_label_concat = np.arange(0,4800, 2.34375)
fig6, axs6 = plt.subplots(4, 1)

#T14R14 T14R9
axs6[0].plot(x2_label_concat, r14_as_85_aenu, label='retest14')
axs6[0].plot(x2_label_concat, t14_as_85_aenu, label='test14')
axs6[0].legend(loc='upper right')
axs6[0].set_title('s14_as_85_aenu (0.832793, rank#3)')
axs6[0].set_xlim(0,1300) # 0 to 1300 Hz

axs6[1].plot(x2_label_concat, r9_as_85_aenu, label='retest9')
axs6[1].plot(x2_label_concat, t14_as_85_aenu, label='test14')
axs6[1].legend(loc='upper right')
axs6[1].set_title('r9_as_85_aenu vs t14_as_85_aenu (0.835228, rank#1)')
axs6[1].set_xlim(0,1300) # 0 to 1300 Hz

# T15R15 T15R1
axs6[2].plot(x2_label_concat, r15_as_85_aenu, label='retest15')
axs6[2].plot(x2_label_concat, t15_as_85_aenu, label='test15')
axs6[2].legend(loc='upper right')
axs6[2].set_title('s15_as_85_aenu (0.845387, rank#4)')
axs6[2].set_xlim(0,1300) # 0 to 1300 Hz

axs6[3].plot(x2_label_concat, r1_as_85_aenu, label='retest1')
axs6[3].plot(x2_label_concat, t15_as_85_aenu, label='test15')
axs6[3].legend(loc='upper right')
axs6[3].set_title('r1_as_85_aenu vs. t15_as_85_aenu (0.89756, rank#1)')
axs6[3].set_xlim(0,1300) # 0 to 1300 Hz

plt.setp(axs6[0].get_xticklabels(), visible=False)
plt.setp(axs6[1].get_xticklabels(), visible=False)
plt.setp(axs6[2].get_xticklabels(), visible=False)
plt.show()

# Subject 20 21
fig7, axs7 = plt.subplots(4, 1)
# T20R20 T20R1
axs7[0].plot(x2_label_concat, r20_as_85_aenu, label='retest20')
axs7[0].plot(x2_label_concat, t20_as_85_aenu, label='test20')
axs7[0].legend(loc='upper right')
axs7[0].set_title('s20_as_85_aenu (0.844889, rank#7)')
axs7[0].set_xlim(0,1300) # 0 to 1300 Hz

axs7[1].plot(x2_label_concat, r1_as_85_aenu, label='retest1')
axs7[1].plot(x2_label_concat, t20_as_85_aenu, label='test20')
axs7[1].legend(loc='upper right')
axs7[1].set_title('r1_as_85_aenu vs t20_as_85_aenu (0.905874, rank#1)')
axs7[1].set_xlim(0,1300) # 0 to 1300 Hz
# T21R21 T21R7
axs7[2].plot(x2_label_concat, r21_as_85_aenu, label='retest21')
axs7[2].plot(x2_label_concat, t21_as_85_aenu, label='test21')
axs7[2].legend(loc='upper right')
axs7[2].set_title('s21_as_85_aenu (0.815956, rank#2)')
axs7[2].set_xlim(0,1300) # 0 to 1300 Hz

axs7[3].plot(x2_label_concat, r7_as_85_aenu, label='retest7')
axs7[3].plot(x2_label_concat, t21_as_85_aenu, label='test21')
axs7[3].legend(loc='upper right')
axs7[3].set_title('r7_as_85_aenu vs. t21_as_85_aenu (0.820878, rank#1)')
axs7[3].set_xlim(0,1300) # 0 to 1300 Hz

plt.setp(axs7[0].get_xticklabels(), visible=False)
plt.setp(axs7[1].get_xticklabels(), visible=False)
plt.setp(axs7[2].get_xticklabels(), visible=False)
plt.show()









# subject
'''
# Correlation Matrix

# EFR
correlation_matrix(corr_EFR_avg_85_a, 'cross correlation of 85dB a_vowel in time domain')
correlation_matrix(corr_EFR_avg_85_e, 'cross correlation of 85dB e_vowel in time domain')
correlation_matrix(corr_EFR_avg_85_n, 'cross correlation of 85dB n_vowel in time domain')
correlation_matrix(corr_EFR_avg_85_u, 'cross correlation of 85dB u_vowel in time domain')

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

# EFR
correlation_matrix_01(corr_EFR_avg_85_a, 'cross correlation of 85dB a_vowel in time domain')
#correlation_matrix_tt_01(corr_EFR_avg_85_a_t, 'cross correlation of 85dB a_vowel in time domain')
#correlation_matrix_rr_01(corr_EFR_avg_85_a_re, 'cross correlation of 85dB a_vowel in time domain')

correlation_matrix_01(corr_EFR_avg_85_e, 'cross correlation of 85dB e_vowel in time domain')
#correlation_matrix_tt_01(corr_EFR_avg_85_e_t, 'cross correlation of 85dB e_vowel in time domain')
#correlation_matrix_rr_01(corr_EFR_avg_85_e_re, 'cross correlation of 85dB e_vowel in time domain')

correlation_matrix_01(corr_EFR_avg_85_n, 'cross correlation of 85dB n_vowel in time domain')
#correlation_matrix_tt_01(corr_EFR_avg_85_n_t, 'cross correlation of 85dB n_vowel in time domain')
#correlation_matrix_rr_01(corr_EFR_avg_85_n_re, 'cross correlation of 85dB n_vowel in time domain')

correlation_matrix_01(corr_EFR_avg_85_u, 'cross correlation of 85dB u_vowel in time domain')
#correlation_matrix_tt_01(corr_EFR_avg_85_u_t, 'cross correlation of 85dB u_vowel in time domain')
#correlation_matrix_rr_01(corr_EFR_avg_85_u_re, 'cross correlation of 85dB u_vowel in time domain')


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

# Correlation Matrix_both



# EFR
'''
correlation_matrix_comb(corr_EFR_avg_85_a, 'cross correlation of 85dB a_vowel in time domain')
correlation_matrix_comb(corr_EFR_avg_85_e, 'cross correlation of 85dB e_vowel in time domain')
correlation_matrix_comb(corr_EFR_avg_85_n, 'cross correlation of 85dB n_vowel in time domain')
correlation_matrix_comb(corr_EFR_avg_85_u, 'cross correlation of 85dB u_vowel in time domain')
'''

# figure 1
#correlation_matrix_comb(corr_EFR_avg_85_aenu, 'cross correlation of 85dB aenu in time domain')

#correlation_matrix_comb(corr_EFR_avg_85_aenu_shrink_03_1, 'cross correlation of shrinked(0.3, 1) 85dB aenu in time domain')
#correlation_matrix_comb(corr_EFR_avg_85_aenu_shrink_05_1, 'cross correlation of shrinked(0.5, 1) 85dB aenu in time domain')


# AS
'''
correlation_matrix_comb(corr_as_win_85_a, 'cross correlation of 85dB a_vowel in frequency domain')
correlation_matrix_comb(corr_as_win_85_e, 'cross correlation of 85dB e_vowel in frequency domain')
correlation_matrix_comb(corr_as_win_85_n, 'cross correlation of 85dB n_vowel in frequency domain')
correlation_matrix_comb(corr_as_win_85_u, 'cross correlation of 85dB u_vowel in frequency domain')
'''
#correlation_matrix_comb(corr_as_85_aenu_shrink_03_1, 'cross correlation of shrinked(0.3, 1) 85dB aenu in frequency domain')
#correlation_matrix_comb(corr_as_85_aenu_shrink_05_1, 'cross correlation of shrinked(0.5, 1) 85dB aenu in frequency domain')


# sum of EFR and AS
#correlation_matrix_comb(corr_sum_avg_85_aenu, 'cross correlation of sum 85dB aenu in time and freq domain')



# AS7
'''
correlation_matrix_comb(corr_as7_85_a, 'cross correlation of 85dB a_vowel in frequency domain 7')
correlation_matrix_comb(corr_as7_85_e, 'cross correlation of 85dB e_vowel in frequency domain 7')
correlation_matrix_comb(corr_as7_85_n, 'cross correlation of 85dB n_vowel in frequency domain 7')
correlation_matrix_comb(corr_as7_85_u, 'cross correlation of 85dB u_vowel in frequency domain 7')
'''







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

