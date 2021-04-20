#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 17:02:59 2018

@author: bruce
"""

# last version = plot_corr_mx_concate_time_linux_v1.6.0.py

import pandas as pd
import numpy as np
from scipy import fftpack
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import librosa



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


#################################


f_dB = lambda x : 20 * np.log10(np.abs(x))


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


# Time domain

# Define window function
win_kaiser = signal.kaiser(1024, beta=14)
win_hamming = signal.hamming(1024)

# average the df_EFR

df_EFR_win = pd.DataFrame()
# implement the window function
for i in range(1408):
    temp_EFR_window = pd.DataFrame((df_EFR.iloc[i,:1024] * win_hamming).values.reshape(1,1024))
    temp_EFR_label = pd.DataFrame(df_EFR.iloc[i, 1024:1031].values.reshape(1,7))
    df_EFR_win = df_EFR_win.append(pd.concat([temp_EFR_window, temp_EFR_label], axis=1, ignore_index=True))
    
# set the title of columns
# df_EFR_avg.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
# df_EFR_avg = df_EFR_avg.sort_values(by=["Condition", "Subject"])
# df_EFR_avg = df_EFR_avg.reset_index(drop=True)
df_EFR_win.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_EFR_win = df_EFR_win.sort_values(by=["Condition", "Subject"])
df_EFR_win = df_EFR_win.reset_index(drop=True)


# average all the subjects , test and retest and keep one sound levels
# filter by 'a vowel and 85Db'
df_EFR_sorted = df_EFR.sort_values(by=["Sound Level", "Vowel","Condition", "Subject"])
df_EFR_sorted = df_EFR_sorted.reset_index(drop=True)
df_EFR_win_sorted = df_EFR_win.sort_values(by=["Sound Level", "Vowel","Condition", "Subject"])
df_EFR_win_sorted = df_EFR_win_sorted.reset_index(drop=True)

# filter55 65 75 sound levels and keep 85dB
# keep vowel condition and subject
df_EFR_85 = pd.DataFrame(df_EFR_sorted.iloc[1056:, :])
df_EFR_85 = df_EFR_85.reset_index(drop=True)
df_EFR_win_85 = pd.DataFrame(df_EFR_win_sorted.iloc[1056:, :])
df_EFR_win_85 = df_EFR_win_85.reset_index(drop=True)

 
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
df_EFR_mid = pd.DataFrame(np.zeros((1408, 95036)))
df_EFR_withzero = pd.concat([df_EFR_data, df_EFR_mid, df_EFR_label], axis=1)
# rename columns
df_EFR_withzero.columns = np.append(np.arange(96060), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])

# for df_EFR_avg_85
df_EFR_85_data = df_EFR_85.iloc[:, :1024]
df_EFR_85_label = df_EFR_85.iloc[:, 1024:]
df_EFR_85_mid = pd.DataFrame(np.zeros((352, 18698)))
df_EFR_85_withzero = pd.concat([df_EFR_85_data, df_EFR_85_mid, df_EFR_85_label], axis=1)
df_EFR_85_withzero.columns = np.append(np.arange(19722), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])


# normalization

# normalize the dataframe by standard deviation
df_EFR_85_data_std = df_EFR_85_data.std(axis=1)
df_EFR_85_data_norm_std = (df_EFR_85_data.iloc[:, :1024]).div(df_EFR_85_data_std, axis=0)


df_EFR_85_vsc = pd.concat([df_EFR_85_data, df_EFR_85_label], axis=1).sort_values(by=["Vowel", "Subject", "Condition"]).reset_index(drop=True)
df_EFR_85_vsc_norm_std = pd.concat([df_EFR_85_data_norm_std, df_EFR_85_label], axis=1).sort_values(by=["Vowel", "Subject", "Condition"]).reset_index(drop=True)
df_EFR_85_withzero_vsc = df_EFR_85_withzero.sort_values(by=["Vowel", "Subject", "Condition"]).reset_index(drop=True)
df_EFR_85_withzero_cvs = df_EFR_85_withzero.sort_values(by=["Condition", "Vowel", "Subject"]).reset_index(drop=True)

df_EFR_85_withzero_cvs_r = df_EFR_85_withzero_cvs.iloc[0:176, :].reset_index(drop=True)
df_EFR_85_withzero_cvs_t = df_EFR_85_withzero_cvs.iloc[176:352, :].reset_index(drop=True)

df_EFR_85_vsc_a = df_EFR_85_vsc.iloc[0:88, :1024]
df_EFR_85_vsc_e = df_EFR_85_vsc.iloc[88:176, :1024].reset_index(drop=True)
df_EFR_85_vsc_n = df_EFR_85_vsc.iloc[176:264, :1024].reset_index(drop=True)
df_EFR_85_vsc_u = df_EFR_85_vsc.iloc[264:352, :1024].reset_index(drop=True)

df_EFR_85_vsc_norm_std_a = df_EFR_85_vsc_norm_std.iloc[0:88, :1024]
df_EFR_85_vsc_norm_std_e = df_EFR_85_vsc_norm_std.iloc[88:176, :1024]
df_EFR_85_vsc_norm_std_n = df_EFR_85_vsc_norm_std.iloc[176:264, :1024]
df_EFR_85_vsc_norm_std_u = df_EFR_85_vsc_norm_std.iloc[264:352, :1024]

df_EFR_85_withzero_vsc_a = df_EFR_85_withzero_vsc.iloc[0:88, :19722]
df_EFR_85_withzero_vsc_e = df_EFR_85_withzero_vsc.iloc[88:176, :19722]
df_EFR_85_withzero_vsc_n = df_EFR_85_withzero_vsc.iloc[176:264, :19722]
df_EFR_85_withzero_vsc_u = df_EFR_85_withzero_vsc.iloc[264:352, :19722]

df_EFR_85_withzero_cvs_r_a = df_EFR_85_withzero_cvs_r.iloc[0:44, :19722]
df_EFR_85_withzero_cvs_r_a_label = df_EFR_85_withzero_cvs_r.iloc[0:44, 19722:]
df_EFR_85_withzero_cvs_r_e = df_EFR_85_withzero_cvs_r.iloc[44:88, :19722]
df_EFR_85_withzero_cvs_r_n = df_EFR_85_withzero_cvs_r.iloc[88:132, :19722]
df_EFR_85_withzero_cvs_r_u = df_EFR_85_withzero_cvs_r.iloc[132:176, :19722]

df_EFR_85_withzero_cvs_t_a = df_EFR_85_withzero_cvs_t.iloc[0:44, :19722]
df_EFR_85_withzero_cvs_t_e = df_EFR_85_withzero_cvs_t.iloc[44:88, :19722]
df_EFR_85_withzero_cvs_t_n = df_EFR_85_withzero_cvs_t.iloc[88:132, :19722]
df_EFR_85_withzero_cvs_t_u = df_EFR_85_withzero_cvs_t.iloc[132:176, :19722]


# concatenate AENU
temp1 = pd.concat([df_EFR_85_vsc_a,df_EFR_85_vsc_e], axis=1)
temp2 = pd.concat([df_EFR_85_vsc_n, df_EFR_85_vsc_u], axis=1)
df_EFR_85_vsc_aenu = pd.concat([temp1, temp2], axis=1, ignore_index=True)


# signal will be used for concatenate aenu is 
# df_EFR_85_vsc_aenu
# 88 sets of signal

'''
# Calculate correlation
# EFR
corr_EFR_avg_85_a = df_EFR_avg_85.iloc[0:44, 0:1024].T.corr(method='pearson').iloc[22:44, 0:22]
corr_EFR_avg_85_e = df_EFR_avg_85.iloc[44:88, 0:1024].T.corr(method='pearson').iloc[22:44, 0:22]
corr_EFR_avg_85_n = df_EFR_avg_85.iloc[88:132, 0:1024].T.corr(method='pearson').iloc[22:44, 0:22]
corr_EFR_avg_85_u = df_EFR_avg_85.iloc[132:176, 0:1024].T.corr(method='pearson').iloc[22:44, 0:22]

corr_EFR_avg_85_aenu = df_EFR_avg_85_aenu.iloc[:, 0:4096].T.corr(method='pearson').iloc[22:44, 0:22]
'''

# test for plot mfcc
signal_in = np.array(df_EFR_85_vsc_aenu.iloc[0, :], dtype=float)
mfccs = librosa.feature.mfcc(y=signal_in, sr=9606, n_mfcc=13)

# visualize the mfcc series
import librosa.display
plt.figure()
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()