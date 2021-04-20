#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 22:33:07 2018

@author: bruce
"""

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
import os

# set saving path
path_result_freq = "/home/bruce/Dropbox/Project/5.Result/5.Result_Nov/2.freq_domain/"

def correlation_matrix(corr_mx, cm_title):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #cmap = cm.get_cmap('jet', 30)
    cax = ax1.matshow(corr_mx, cmap='gray')
    #cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    fig.colorbar(cax)
    ax1.grid(False)
    plt.title(cm_title)
    #plt.title('cross correlation of test and retest')
    ylabels = ['T1','T2','T3','T4','T6','T7','T8','T9', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T25']
    xlabels = ['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
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
    # otherwise it is not working
    temp = np.asarray(corr_mx)
    output = (temp == temp.max(axis=1)[:,None]) # along rows

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #cmap = cm.get_cmap('jet', 30)
    cax = ax1.matshow(output, cmap='binary')
    # cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    #fig.colorbar(cax)
    ax1.grid(False)
    plt.title(cm_title)
    ylabels=['T1','T2','T3','T4','T6','T7','T8','T9', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T25']
    xlabels=['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
    ax1.set_xticks(np.arange(len(xlabels)))
    ax1.set_yticks(np.arange(len(ylabels)))
    ax1.set_xticklabels(xlabels,fontsize=6)
    ax1.set_yticklabels(ylabels,fontsize=6)
    plt.show()
    

def correlation_matrix_min_01_comb(corr_mx1 ,corr_mx2, cm_title1, cm_title2):
    # find the minimum in each row 
    # input corr_mx is a dataframe
    # need to convert it into a array first
    # otherwise it is not working
    
    temp = np.asarray(corr_mx1)
    output1 = (temp == temp.min(axis=1)[:,None]) # along rows
    temp = np.asarray(corr_mx2)
    output2 = (temp == temp.min(axis=1)[:,None]) # along rows
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # figure 1
    im1 = ax1.matshow(output1, cmap='binary')
    #fig.colorbar(im1, ax1)
    ax1.grid(False)
    ax1.set_title(cm_title1)
    ylabels=['T1','T2','T3','T4','T6','T7','T8','T9', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T25']
    xlabels=['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
    ax1.set_xticks(np.arange(len(xlabels)))
    ax1.set_yticks(np.arange(len(ylabels)))
    ax1.set_xticklabels(xlabels,fontsize=6)
    ax1.set_yticklabels(ylabels,fontsize=6)
    
    # figure 2 
    im2 = ax2.matshow(output2, cmap='binary')
    #fig.colorbar(im2, ax2)
    ax2.grid(False)
    ax2.set_title(cm_title2)
    ylabels=['T1','T2','T3','T4','T6','T7','T8','T9', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T25']
    xlabels=['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
    ax2.set_xticks(np.arange(len(xlabels)))
    ax2.set_yticks(np.arange(len(ylabels)))
    ax2.set_xticklabels(xlabels,fontsize=6)
    ax2.set_yticklabels(ylabels,fontsize=6)
    plt.show()
    
    
def correlation_matrix_tt_01(corr_mx, cm_title):
    # find the maximum in each row 

    # input corr_mx is a dataframe
    # need to convert it into a array first
    # otherwise it is not working
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
    ax1.set_xticklabels(xlabels, fontsize=6)
    ax1.set_yticklabels(ylabels, fontsize=6)
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
    # cmap = cm.get_cmap('jet', 30)
    cax = ax1.matshow(output, cmap='gray')
    # cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
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


# eg: plot_mag_db(df_as_85_vsc, 1, "Subject")
def fig_mag_db(signal_in, subject_number = 'subject_number', title = 'title', filename = 'filename'):
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(signal_in.iloc[2*(subject_number-1), :48030], '-')
    plt.plot(signal_in.iloc[2*(subject_number-1)+1, :48030], '-')
    plt.ylabel('magnitude')
    plt.xlim(0,10000)
    plt.legend(('Retest', 'Test'), loc='upper right')
    plt.title(title)
    # plt.subplot(2,1,2)
    # plt.plot(signal_in.iloc[2*(subject_number-1), :48030].apply(f_dB), '-')
    # plt.plot(signal_in.iloc[2*(subject_number-1)+1, :48030].apply(f_dB), '-')
    # plt.xlabel('Frequency(Hz)')
    # plt.ylabel('dB')
    # plt.xlim(0,10000)
    # plt.legend(('Retest', 'Test'), loc='lower right')
    plt.show()
    plt.savefig(filename)
    
def fig_mag_in_1(signal_in, title = 'title', path = 'path', filename = 'filename'):
    plt.figure()
    sub_title = ['1', '2', '3', '4', '6', '7', '8', '9', '11', '12',\
                 '13', '14', '15', '16', '17', '18', '19', '20', '21',\
                 '22', '23', '25']
    for i in range(22):
        plt.subplot(11,2,i+1)
        x_label = np.arange(0, 4803, 0.1)
        plt.plot(x_label, signal_in.iloc[2*i, :48030], '-')
        plt.plot(x_label, signal_in.iloc[2*i+1, :48030], '-')
        plt.ylabel(sub_title[i])
        plt.xlim(0,1000)
        plt.legend(('Retest', 'Test'), loc='upper right', fontsize='xx-small')
    plt.suptitle(title) # add a centered title to the figure
    plt.show()
    plt.savefig(os.path.join(path, filename), dpi=300)
    

def fig_test_in_1(signal_in_1, signal_in_2, title = 'title', path = 'path', filename = 'filename'):
    plt.figure()
    sub_title = ['1', '2', '3', '4', '6', '7', '8', '9', '11', '12',\
                 '13', '14', '15', '16', '17', '18', '19', '20', '21',\
                 '22', '23', '25']
    for i in range(22):
        plt.subplot(11,2,i+1)
        x_label = np.arange(0, 4803, 0.1)
        plt.plot(x_label, signal_in_1.iloc[2*i, :48030], '-')
        plt.plot(x_label, signal_in_2.iloc[2*i, :48030], '-')
        plt.ylabel(sub_title[i])
        plt.xlim(0,1000)
        plt.legend(('no window', 'window'), loc='upper right', fontsize='xx-small')
    plt.suptitle(title) # add a centered title to the figure
    plt.show()
    plt.savefig(os.path.join(path, filename), dpi=300)


def fig_retest_in_1(signal_in_1, signal_in_2, title = 'title', path = 'path', filename = 'filename'):
    plt.figure()
    sub_title = ['1', '2', '3', '4', '6', '7', '8', '9', '11', '12',\
                 '13', '14', '15', '16', '17', '18', '19', '20', '21',\
                 '22', '23', '25']
    for i in range(22):
        plt.subplot(11,2,i+1)
        x_label = np.arange(0, 4803, 0.1)
        plt.plot(x_label, signal_in_1.iloc[2*i+1, :48030], '-')
        plt.plot(x_label, signal_in_2.iloc[2*i+1, :48030], '-')
        plt.ylabel(sub_title[i])
        plt.xlim(0,1000)
        plt.legend(('no window', 'window'), loc='upper right', fontsize='xx-small')
    plt.suptitle(title) # add a centered title to the figure
    plt.show()
    plt.savefig(os.path.join(path, filename), dpi=300)


def distance_mx(sig_in):
    # freq_range -> from 0 to ???
    freq_range = 13000
    matrix_temp = np.zeros((22, 22))
    matrix_temp_square = np.zeros((22, 22))
    for i in range(22):
        for j in range(22):
            temp = np.asarray(sig_in.iloc[2*i, 0:freq_range] - sig_in.iloc[2*j+1, 0:freq_range])
            temp_sum = 0
            temp_square_sum = 0
            for k in range(freq_range):
                #test_t3 = (abs(temp_series[k]))**2
                #print(test_t3)
                temp_sum = temp_sum + abs(temp[k])
                temp_square_sum = temp_square_sum + (abs(temp[k]))**2
            matrix_temp[i][j] = temp_sum
            matrix_temp_square[i][j] = temp_square_sum
    output_1 = pd.DataFrame(matrix_temp)
    output_2 = pd.DataFrame(matrix_temp_square)
    # output 1 is similar with euclidian diatance eg. x1+jy1 -> sqrt(x1**2 + y1**2)
    # output 1 is square result eg. x1+jy1 -> x1**2 + y1**2
    return output_1, output_2


def complex_coherence_mx(input_signal):
    # compute the magnitude squared coherence based on signal.coherence
    # then create the matrix with values
    # higher value -> better coherence value
    sig_in = input_signal.copy()
    matrix_temp = np.zeros((22, 22))
    for i in range(22):
        for j in range(22):
            # temp here is the
            temp_sum = 0
            sig_in_1 = np.array(sig_in.iloc[2*i, :])
            sig_in_2 = np.array(sig_in.iloc[2*j+1, :])
            # signal 9606Hz length 106.6ms window length 10ms -> nperseg=96
            f, temp_Cxy = signal.coherence(sig_in_1, sig_in_2, fs=9606, nperseg=192)
            
            # delete values lower than 0.01
            for l in range(len(temp_Cxy)):
                if temp_Cxy[l] < 0.1:
                    temp_Cxy[l] = 0
            # delete finish
            
            # test
            '''
            if i ==0 and j == 0:
                plt.figure()
                plt.semilogy(f, temp_Cxy)
                plt.title("test in complex_coherence_mx")
                plt.show()
            '''
            # test finish
            for k in range(len(temp_Cxy)):
                #test_t3 = (abs(temp_series[k]))**2
                #print(test_t3)
                temp_sum = temp_sum + abs(temp_Cxy[k])
            matrix_temp[i][j] = temp_sum
    output_3 = pd.DataFrame(matrix_temp)
    return output_3



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
df_EFR_avg_85_withzero.columns = np.append(np.arange(96060), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])


# normalization

# normalize the dataframe by standard deviation
df_EFR_avg_85_data_std = df_EFR_avg_85_data.std(axis=1)
df_EFR_avg_85_data_norm_std = (df_EFR_avg_85_data.iloc[:, :1024]).div(df_EFR_avg_85_data_std, axis=0)
# normalize the dataframe by 
df_EFR_avg_85_data_sumofsquare = (np.square(df_EFR_avg_85_data.iloc[:, :1024])).sum(axis=1)
df_EFR_avg_85_data_norm_ss = (df_EFR_avg_85_data.iloc[:, :1024]).div(df_EFR_avg_85_data_sumofsquare, axis=0)


df_EFR_avg_85_vsc = pd.concat([df_EFR_avg_85_data, df_EFR_avg_85_label], axis=1).sort_values(by=["Vowel", "Subject", "Condition"]).reset_index(drop=True)
df_EFR_avg_85_vsc_norm_std = pd.concat([df_EFR_avg_85_data_norm_std, df_EFR_avg_85_label], axis=1).sort_values(by=["Vowel", "Subject", "Condition"]).reset_index(drop=True)
df_EFR_avg_85_vsc_norm_ss = pd.concat([df_EFR_avg_85_data_norm_ss, df_EFR_avg_85_label], axis=1).sort_values(by=["Vowel", "Subject", "Condition"]).reset_index(drop=True)
df_EFR_avg_85_withzero_vsc = df_EFR_avg_85_withzero.sort_values(by=["Vowel", "Subject", "Condition"]).reset_index(drop=True)


df_EFR_avg_85_vsc_a = df_EFR_avg_85_vsc.iloc[0:44, :1024]
df_EFR_avg_85_vsc_e = df_EFR_avg_85_vsc.iloc[44:88, :1024]
df_EFR_avg_85_vsc_n = df_EFR_avg_85_vsc.iloc[88:132, :1024]
df_EFR_avg_85_vsc_u = df_EFR_avg_85_vsc.iloc[132:176, :1024]

df_EFR_avg_85_vsc_norm_std_a = df_EFR_avg_85_vsc_norm_std.iloc[0:44, :1024]
df_EFR_avg_85_vsc_norm_std_e = df_EFR_avg_85_vsc_norm_std.iloc[44:88, :1024]
df_EFR_avg_85_vsc_norm_std_n = df_EFR_avg_85_vsc_norm_std.iloc[88:132, :1024]
df_EFR_avg_85_vsc_norm_std_u = df_EFR_avg_85_vsc_norm_std.iloc[132:176, :1024]

df_EFR_avg_85_vsc_norm_ss_a = df_EFR_avg_85_vsc_norm_ss.iloc[0:44, :1024]
df_EFR_avg_85_vsc_norm_ss_e = df_EFR_avg_85_vsc_norm_ss.iloc[44:88, :1024]
df_EFR_avg_85_vsc_norm_ss_n = df_EFR_avg_85_vsc_norm_ss.iloc[88:132, :1024]
df_EFR_avg_85_vsc_norm_ss_u = df_EFR_avg_85_vsc_norm_ss.iloc[132:176, :1024]

df_EFR_avg_85_withzero_vsc_a = df_EFR_avg_85_withzero_vsc.iloc[0:44, :48030]
df_EFR_avg_85_withzero_vsc_e = df_EFR_avg_85_withzero_vsc.iloc[44:88, :48030]
df_EFR_avg_85_withzero_vsc_n = df_EFR_avg_85_withzero_vsc.iloc[88:132, :48030]
df_EFR_avg_85_withzero_vsc_u = df_EFR_avg_85_withzero_vsc.iloc[132:176, :48030]

# concatenate AENU
temp1 = pd.concat([df_EFR_avg_85_vsc.iloc[0:44, 0:1024].reset_index(drop=True),df_EFR_avg_85_vsc.iloc[44:88, 0:1024].reset_index(drop=True)], axis=1)
temp2 = pd.concat([df_EFR_avg_85_vsc.iloc[88:132, 0:1024].reset_index(drop=True), df_EFR_avg_85_vsc.iloc[132:176, 0:1024].reset_index(drop=True)], axis=1)
df_EFR_avg_85_aenu = pd.concat([temp1, temp2], axis=1, ignore_index=True)
#df_EFR_avg_85_aenu_withzero = pd.concat([df_EFR_avg_85_aenu, pd.DataFrame(np.zeros((44, 36864)))] , axis=1)


# df_EFR_avg_win_85
df_EFR_avg_win_85_data = df_EFR_avg_win_85.iloc[:, :1024]
df_EFR_avg_win_85_label = df_EFR_avg_win_85.iloc[:, 1024:]
df_EFR_avg_win_85_mid = pd.DataFrame(np.zeros((176, 95036)))
df_EFR_avg_win_85_withzero = pd.concat([df_EFR_avg_win_85_data, df_EFR_avg_win_85_mid, df_EFR_avg_win_85_label], axis=1)
df_EFR_avg_win_85_withzero.columns = np.append(np.arange(96060), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])



# test

sig_in = df_EFR_avg_85_vsc_a.copy()
# sig_in = df_EFR_avg_85_vsc_a.copy()
# freq_range -> from 0 to ???

matrix_temp = np.zeros((22, 22))
for i in range(22):
    for j in range(22):
        # temp here is the 
        temp_sum = 0
        sig_in_1 = np.array(sig_in.iloc[2*i, :]) 
        sig_in_2 = np.array(sig_in.iloc[2*j+1, :])
        f, temp_Cxy = signal.coherence(sig_in_1, sig_in_2, fs = 9606, nperseg=32)
        
        # select 0 to 1300Hz 
        '''
        signal_length = 160
        f = f[0:signal_length]
        temp_Cxy = temp_Cxy[0:signal_length]
        '''
        
        # delete values lower than threshold(min_Cxy)
        
        min_Cxy = 0.1
        for l in range(len(temp_Cxy)):
            if temp_Cxy[l] < min_Cxy:
                temp_Cxy[l] = 0
        
        # delete finish
        
        # test
        if i == 3 and j == 3:
            plt.figure()
            plt.semilogy(f, temp_Cxy)
            test_f = f
            test_temp_Cxy = temp_Cxy
            plt.show()
        # test finish
        
        for k in range(len(temp_Cxy)):
            #test_t3 = (abs(temp_series[k]))**2
            #print(test_t3)
            temp_sum = temp_sum + abs(temp_Cxy[k])
        matrix_temp[i][j] = temp_sum
output_3 = pd.DataFrame(matrix_temp)
correlation_matrix_01(output_3, "complex coherence for 85dB a vowel(test)")

# test finish


#  magnitude squared coherence
df_coherence_avg_85_vsc_a = complex_coherence_mx(df_EFR_avg_85_vsc_a)
df_coherence_avg_85_vsc_e = complex_coherence_mx(df_EFR_avg_85_vsc_e)
df_coherence_avg_85_vsc_n = complex_coherence_mx(df_EFR_avg_85_vsc_n)
df_coherence_avg_85_vsc_u = complex_coherence_mx(df_EFR_avg_85_vsc_u)

df_coherence_avg_85_vsc_norm_std_a = complex_coherence_mx(df_EFR_avg_85_vsc_norm_std_a)
df_coherence_avg_85_vsc_norm_std_e = complex_coherence_mx(df_EFR_avg_85_vsc_norm_std_e)
df_coherence_avg_85_vsc_norm_std_n = complex_coherence_mx(df_EFR_avg_85_vsc_norm_std_n)
df_coherence_avg_85_vsc_norm_std_u = complex_coherence_mx(df_EFR_avg_85_vsc_norm_std_u)

df_coherence_avg_85_vsc_norm_ss_a = complex_coherence_mx(df_EFR_avg_85_vsc_norm_ss_a)
df_coherence_avg_85_vsc_norm_ss_e = complex_coherence_mx(df_EFR_avg_85_vsc_norm_ss_e)
df_coherence_avg_85_vsc_norm_ss_n = complex_coherence_mx(df_EFR_avg_85_vsc_norm_ss_n)
df_coherence_avg_85_vsc_norm_ss_u = complex_coherence_mx(df_EFR_avg_85_vsc_norm_ss_u)

# sum of aenu
df_coherence_avg_85_vsc_ae = df_coherence_avg_85_vsc_a.add(df_coherence_avg_85_vsc_e)
df_coherence_avg_85_vsc_nu = df_coherence_avg_85_vsc_n.add(df_coherence_avg_85_vsc_u)
df_coherence_avg_85_vsc_aenu = df_coherence_avg_85_vsc_ae.add(df_coherence_avg_85_vsc_nu)
# df_coherence_avg_85_0_vsc_a = complex_coherence_mx(df_EFR_avg_85_withzero_vsc_a)

# plot the coherence matrix
correlation_matrix_01(df_coherence_avg_85_vsc_a, "complex coherence for 85dB a vowel")
correlation_matrix_01(df_coherence_avg_85_vsc_e, "complex coherence for 85dB e vowel")
correlation_matrix_01(df_coherence_avg_85_vsc_n, "complex coherence for 85dB n vowel")
correlation_matrix_01(df_coherence_avg_85_vsc_u, "complex coherence for 85dB u vowel")
correlation_matrix_01(df_coherence_avg_85_vsc_aenu, "complex coherence for 85dB concatenate aenu ")
# correlation_matrix_01(df_coherence_avg_85_0_vsc_a, "complex coherence for 85dB a vowel (zero padding)")

correlation_matrix_01(df_coherence_avg_85_vsc_norm_std_a, "complex coherence for 85dB a vowel(std)")
correlation_matrix_01(df_coherence_avg_85_vsc_norm_std_e, "complex coherence for 85dB e vowel(std)")
correlation_matrix_01(df_coherence_avg_85_vsc_norm_std_n, "complex coherence for 85dB n vowel(std)")
correlation_matrix_01(df_coherence_avg_85_vsc_norm_std_u, "complex coherence for 85dB u vowel(std)")

correlation_matrix_01(df_coherence_avg_85_vsc_norm_ss_a, "complex coherence for 85dB a vowel(sum of square)")
correlation_matrix_01(df_coherence_avg_85_vsc_norm_ss_e, "complex coherence for 85dB e vowel(sum of square)")
correlation_matrix_01(df_coherence_avg_85_vsc_norm_ss_n, "complex coherence for 85dB n vowel(sum of square)")
correlation_matrix_01(df_coherence_avg_85_vsc_norm_ss_u, "complex coherence for 85dB u vowel(sum of square)")


'''
# test##############
# test(detrend)
temp_test = np.asarray(df_EFR.iloc[0, 0:1024])
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

"""
# create a new dataframe with zero-padding amplitude spectrum

'''
# for df_EFR
df_as_7= pd.DataFrame()
for i in range(1408):
    temp_EFR = df_EFR_withzero.iloc[i, 0:9606]
    temp_as = np.abs((fftpack.fft(temp_EFR)/n2)[range(int(n2/2))])
    #df_as_7 = pd.concat([df_as_7, temp_as_7_t], axis=0)
    df_as_7 = df_as_7.append(pd.DataFrame(np.array([temp_as[100], temp_as[200], temp_as[300], temp_as[400], \
                                                    temp_as[500], temp_as[600], temp_as[700]]).reshape(1,7)), ignore_index = True)
   
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

#resort df_as_85 based on 1.vowel, 2.subject 3.condition
df_as_85_vsc = df_as_85.sort_values(by=["Vowel", "Subject", "Condition"])
df_as_85_vsc = df_as_85_vsc.reset_index(drop=True)
df_as_85_vsc_label = df_as_85_vsc.iloc[:,48030:]
df_as_win_85_vsc = df_as_win_85.sort_values(by=["Vowel", "Subject", "Condition"])
df_as_win_85_vsc = df_as_win_85_vsc.reset_index(drop=True)
df_as_win_85_vsc_label = df_as_win_85_vsc.iloc[:,48030:]


df_as_85_vsc_a = df_as_85_vsc.iloc[0:44, :]
df_as_85_vsc_e = df_as_85_vsc.iloc[44:88, :]
df_as_85_vsc_n = df_as_85_vsc.iloc[88:132, :]
df_as_85_vsc_u = df_as_85_vsc.iloc[132:176, :]
df_as_win_85_vsc_a = df_as_win_85_vsc.iloc[0:44, :]
df_as_win_85_vsc_e = df_as_win_85_vsc.iloc[44:88, :]
df_as_win_85_vsc_n = df_as_win_85_vsc.iloc[88:132, :]
df_as_win_85_vsc_u = df_as_win_85_vsc.iloc[132:176, :]



# test
#fig_mag_db(df_as_85_vsc_a, 1, title = '85dB a vowel Subject 1 in frequency domain', filename = '85_a_s1_f_domain.png')


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


# Complex Value
from scipy.spatial import distance_matrix

df_cx_85= pd.DataFrame()
df_cx_win_85= pd.DataFrame()
for i in range(176):
    #temp_aenu_EFR = df_EFR_avg_aenu_withzero.iloc[i, 0:9606]
    temp_cx = (fftpack.fft(df_EFR_avg_85_withzero.iloc[i, 0:96060])/n2)[range(int(n2/2))]
    df_cx_85 = df_cx_85.append(pd.DataFrame(temp_cx.reshape(1,48030)), ignore_index = True)
    temp_cx_win = (fftpack.fft(df_EFR_avg_win_85_withzero.iloc[i, 0:96060])/n2)[range(int(n2/2))]
    df_cx_win_85 = df_cx_win_85.append(pd.DataFrame(temp_cx_win.reshape(1,48030)), ignore_index = True)

df_cx_85 = pd.concat([df_cx_85, df_EFR_avg_85_label], axis=1) # add labels on it
df_cx_win_85 = pd.concat([df_cx_win_85, df_EFR_avg_win_85_label], axis=1) # add labels on it


#resort df_as_85 based on 1.vowel, 2.subject 3.condition
df_cx_85_vsc = df_cx_85.sort_values(by=["Vowel", "Subject", "Condition"])
df_cx_85_vsc = df_cx_85_vsc.reset_index(drop=True)
df_cx_85_vsc_label = df_cx_85_vsc.iloc[:,48030:]
df_cx_win_85_vsc = df_cx_win_85.sort_values(by=["Vowel", "Subject", "Condition"])
df_cx_win_85_vsc = df_cx_win_85_vsc.reset_index(drop=True)
df_cx_win_85_vsc_label = df_cx_win_85_vsc.iloc[:,48030:]

# normalize df_cx_85_vsc but didn't change the name

df_cx_85_vsc = df_cx_85_vsc.iloc[:, :48030].div((df_cx_85_vsc.iloc[:, :48030].abs()**2).sum(axis=1).div(48030).pow(1/2), axis=1)
df_cx_win_85_vsc = df_cx_win_85_vsc.iloc[:, :48030].div(((np.square(df_cx_win_85_vsc.iloc[:, :48030].abs())).sum(axis=1).div(48030)).pow(1/2), axis=1)

# df_cx_85_vsc = (df_cx_85_vsc.iloc[:, :48030]/(df_cx_85_vsc.iloc[:, :48030].abs().max()-df_cx_85_vsc.iloc[:, :48030].abs().min()))
# df_cx_win_85_vsc = (df_cx_win_85_vsc.iloc[:, :48030]/(df_cx_win_85_vsc.iloc[:, :48030].abs().max()-df_cx_win_85_vsc.iloc[:, :48030].abs().min()))



df_cx_85_vsc_a = df_cx_85_vsc.iloc[0:44, :].reset_index(drop=True)
df_cx_85_vsc_e = df_cx_85_vsc.iloc[44:88, :].reset_index(drop=True)
df_cx_85_vsc_n = df_cx_85_vsc.iloc[88:132, :].reset_index(drop=True)
df_cx_85_vsc_u = df_cx_85_vsc.iloc[132:176, :].reset_index(drop=True)
df_cx_win_85_vsc_a = df_cx_win_85_vsc.iloc[0:44, :].reset_index(drop=True)
df_cx_win_85_vsc_e = df_cx_win_85_vsc.iloc[44:88, :].reset_index(drop=True)
df_cx_win_85_vsc_n = df_cx_win_85_vsc.iloc[88:132, :].reset_index(drop=True)
df_cx_win_85_vsc_u = df_cx_win_85_vsc.iloc[132:176, :].reset_index(drop=True)

# test
# plot for test



# distance matrix

distance_cx_85_vsc_a, distance_cx_85_vsc_a_square = distance_mx(df_cx_85_vsc_a)
distance_cx_85_vsc_e, distance_cx_85_vsc_e_square = distance_mx(df_cx_85_vsc_e)
distance_cx_85_vsc_n, distance_cx_85_vsc_n_square = distance_mx(df_cx_85_vsc_n)
distance_cx_85_vsc_u, distance_cx_85_vsc_u_square = distance_mx(df_cx_85_vsc_u)


# test 
'''
# coherence_cx_85_vsc_a = complex_coherence_mx(df_cx_85_vsc_a)
sig_in = df_cx_85_vsc_a.copy()
# sig_in = df_cx_85_vsc_a.copy().astype('complex256')
freq_range = 13000
matrix_temp = np.zeros((22, 22))



for i in range(22):
    for j in range(22):
        # temp here is the 
        # temp_numerator = pd.DataFrame()
        temp_numerator_1 = sig_in.iloc[[2*i], 0:freq_range].reset_index(drop=True)
        temp_numerator_2 = np.conj(sig_in.iloc[[2*j+1], 0:freq_range]).reset_index(drop=True)
        temp_numerator = (temp_numerator_1 * temp_numerator_2).mean().to_frame().T.abs()
        temp_denominator_1_signal = sig_in.iloc[[2*i], 0:freq_range]
        temp_denominator_1 = np.sqrt((sig_in.iloc[[2*i], 0:freq_range].mul(np.conj(sig_in.iloc[[2*i], 0:freq_range])).reset_index(drop=True)).mean().to_frame().T)
        temp_denominator_1_compare = (sig_in.iloc[[2*i], 0:freq_range].abs()**2).sum(axis=1).mean()# .apply(np.sqrt)
        temp_denominator_2 = np.sqrt(sig_in.iloc[[2*j+1], 0:freq_range].mul(np.conj(sig_in.iloc[[2*j+1], 0:freq_range])).reset_index(drop=True).mean().to_frame().T)
        temp_denominator =  temp_denominator_1 * temp_denominator_2
        temp = temp_numerator.divide(temp_denominator)
        temp_maximum_value = temp
        matrix_temp[i][j] = temp_maximum_value
coherence_cx_85_vsc_a = pd.DataFrame(matrix_temp)
'''
####### 

correlation_matrix_min_01_comb(distance_cx_85_vsc_a,\
                               distance_cx_85_vsc_a_square,\
                               cm_title1='distance_cx_85_vsc_a(smallest) euclidian format',\
                               cm_title2='distance_cx_85_vsc_a(smallest) square')

correlation_matrix_min_01_comb(distance_cx_85_vsc_e,\
                               distance_cx_85_vsc_e_square,\
                               cm_title1='distance_cx_85_vsc_e(smallest) euclidian format',\
                               cm_title2='distance_cx_85_vsc_e(smallest) square')

correlation_matrix_min_01_comb(distance_cx_85_vsc_n,\
                               distance_cx_85_vsc_n_square,\
                               cm_title1='distance_cx_85_vsc_n(smallest) euclidian format',\
                               cm_title2='distance_cx_85_vsc_n(smallest) square')

correlation_matrix_min_01_comb(distance_cx_85_vsc_u,\
                               distance_cx_85_vsc_u_square,\
                               cm_title1='distance_cx_85_vsc_u(smallest) euclidian format',\
                               cm_title2='distance_cx_85_vsc_u(smallest) square')


# Calculate correlation

'''
# EFR
corr_EFR_avg_85_a = df_EFR_avg_85.iloc[0:44, 0:1024].T.corr(method='pearson').iloc[22:44, 0:22]
corr_EFR_avg_85_e = df_EFR_avg_85.iloc[44:88, 0:1024].T.corr(method='pearson').iloc[22:44, 0:22]
corr_EFR_avg_85_n = df_EFR_avg_85.iloc[88:132, 0:1024].T.corr(method='pearson').iloc[22:44, 0:22]
corr_EFR_avg_85_u = df_EFR_avg_85.iloc[132:176, 0:1024].T.corr(method='pearson').iloc[22:44, 0:22]

corr_EFR_avg_85_a_t = df_EFR_avg_85.iloc[0:44, 0:1024].T.corr(method='pearson').iloc[0:22, 0:22]
corr_EFR_avg_85_e_t = df_EFR_avg_85.iloc[44:88, 0:1024].T.corr(method='pearson').iloc[0:22, 0:22]
corr_EFR_avg_85_n_t = df_EFR_avg_85.iloc[88:132, 0:1024].T.corr(method='pearson').iloc[0:22, 0:22]
corr_EFR_avg_85_u_t = df_EFR_avg_85.iloc[132:176, 0:1024].T.corr(method='pearson').iloc[0:22, 0:22]

corr_EFR_avg_85_a_re = df_EFR_avg_85.iloc[0:44, 0:1024].T.corr(method='pearson').iloc[22:44, 22:44]
corr_EFR_avg_85_e_re = df_EFR_avg_85.iloc[44:88, 0:1024].T.corr(method='pearson').iloc[22:44, 22:44]
corr_EFR_avg_85_n_re = df_EFR_avg_85.iloc[88:132, 0:1024].T.corr(method='pearson').iloc[22:44, 22:44]
corr_EFR_avg_85_u_re = df_EFR_avg_85.iloc[132:176, 0:1024].T.corr(method='pearson').iloc[22:44, 22:44]

# AS
corr_as_85_a = df_as_85.iloc[0:44, 0:4803].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as_85_e = df_as_85.iloc[44:88, 0:4803].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as_85_n = df_as_85.iloc[88:132, 0:4803].T.corr(method='pearson').iloc[22:44, 0:22]
corr_as_85_u = df_as_85.iloc[132:176, 0:4803].T.corr(method='pearson').iloc[22:44, 0:22]

corr_as_85_a_t = df_as_85.iloc[0:44, 0:4803].T.corr(method='pearson').iloc[0:22, 0:22]
corr_as_85_e_t = df_as_85.iloc[44:88, 0:4803].T.corr(method='pearson').iloc[0:22, 0:22]
corr_as_85_n_t = df_as_85.iloc[88:132, 0:4803].T.corr(method='pearson').iloc[0:22, 0:22]
corr_as_85_u_t = df_as_85.iloc[132:176, 0:4803].T.corr(method='pearson').iloc[0:22, 0:22]

corr_as_85_a_re = df_as_85.iloc[0:44, 0:4803].T.corr(method='pearson').iloc[22:44, 22:44]
corr_as_85_e_re = df_as_85.iloc[44:88, 0:4803].T.corr(method='pearson').iloc[22:44, 22:44]
corr_as_85_n_re = df_as_85.iloc[88:132, 0:4803].T.corr(method='pearson').iloc[22:44, 22:44]
corr_as_85_u_re = df_as_85.iloc[132:176, 0:4803].T.corr(method='pearson').iloc[22:44, 22:44]

#AS7
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





# plot the frequency domain

# a vowel
'''
fig_mag_db(df_as_85_vsc_a, 1, title = '85dB a vowel Subject 1 in frequency domain', filename = '85_a_s1_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 2, title = '85dB a vowel Subject 2 in frequency domain', filename = '85_a_s2_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 3, title = '85dB a vowel Subject 3 in frequency domain', filename = '85_a_s3_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 4, title = '85dB a vowel Subject 4 in frequency domain', filename = '85_a_s4_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 5, title = '85dB a vowel Subject 5 in frequency domain', filename = '85_a_s5_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 6, title = '85dB a vowel Subject 6 in frequency domain', filename = '85_a_s6_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 7, title = '85dB a vowel Subject 7 in frequency domain', filename = '85_a_s7_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 8, title = '85dB a vowel Subject 8 in frequency domain', filename = '85_a_s8_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 9, title = '85dB a vowel Subject 9 in frequency domain', filename = '85_a_s9_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 10, title = '85dB a vowel Subject 10 in frequency domain', filename = '85_a_s10_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 11, title = '85dB a vowel Subject 11 in frequency domain', filename = '85_a_s11_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 12, title = '85dB a vowel Subject 12 in frequency domain', filename = '85_a_s12_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 13, title = '85dB a vowel Subject 13 in frequency domain', filename = '85_a_s13_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 14, title = '85dB a vowel Subject 14 in frequency domain', filename = '85_a_s14_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 15, title = '85dB a vowel Subject 15 in frequency domain', filename = '85_a_s15_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 16, title = '85dB a vowel Subject 16 in frequency domain', filename = '85_a_s16_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 17, title = '85dB a vowel Subject 17 in frequency domain', filename = '85_a_s17_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 18, title = '85dB a vowel Subject 18 in frequency domain', filename = '85_a_s18_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 19, title = '85dB a vowel Subject 19 in frequency domain', filename = '85_a_s19_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 20, title = '85dB a vowel Subject 20 in frequency domain', filename = '85_a_s20_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 21, title = '85dB a vowel Subject 21 in frequency domain', filename = '85_a_s21_f_domain.png')
fig_mag_db(df_as_85_vsc_a, 22, title = '85dB a vowel Subject 22 in frequency domain', filename = '85_a_s22_f_domain.png')
'''
# plot 22 figures in one figure    
'''
fig_mag_in_1(df_as_win_85_vsc_a, title = '85dB a vowel subjects in frequency domain', \
             path = path_result_freq, \
             filename = '85_a_f_domain.png')

fig_test_in_1(df_as_85_vsc_a, df_as_win_85_vsc_a, title = '85dB a vowel test in frequency domain', \
             path = path_result_freq, \
             filename = '85_a_f_domain_test.png')

fig_retest_in_1(df_as_85_vsc_a, df_as_win_85_vsc_a, title = '85dB a vowel retests in frequency domain', \
             path = path_result_freq, \
             filename = '85_a_f_domain_retest.png')
'''


# e vowel
'''
fig_mag_db(df_as_85_vsc_e, 1, title = '85dB e vowel Subject 1 in frequency domain', filename = '85_e_s1_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 2, title = '85dB e vowel Subject 2 in frequency domain', filename = '85_e_s2_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 3, title = '85dB e vowel Subject 3 in frequency domain', filename = '85_e_s3_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 4, title = '85dB e vowel Subject 4 in frequency domain', filename = '85_e_s4_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 5, title = '85dB e vowel Subject 5 in frequency domain', filename = '85_e_s5_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 6, title = '85dB e vowel Subject 6 in frequency domain', filename = '85_e_s6_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 7, title = '85dB e vowel Subject 7 in frequency domain', filename = '85_e_s7_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 8, title = '85dB e vowel Subject 8 in frequency domain', filename = '85_e_s8_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 9, title = '85dB e vowel Subject 9 in frequency domain', filename = '85_e_s9_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 10, title = '85dB e vowel Subject 10 in frequency domain', filename = '85_e_s10_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 11, title = '85dB e vowel Subject 11 in frequency domain', filename = '85_e_s11_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 12, title = '85dB e vowel Subject 12 in frequency domain', filename = '85_e_s12_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 13, title = '85dB e vowel Subject 13 in frequency domain', filename = '85_e_s13_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 14, title = '85dB e vowel Subject 14 in frequency domain', filename = '85_e_s14_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 15, title = '85dB e vowel Subject 15 in frequency domain', filename = '85_e_s15_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 16, title = '85dB e vowel Subject 16 in frequency domain', filename = '85_e_s16_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 17, title = '85dB e vowel Subject 17 in frequency domain', filename = '85_e_s17_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 18, title = '85dB e vowel Subject 18 in frequency domain', filename = '85_e_s18_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 19, title = '85dB e vowel Subject 19 in frequency domain', filename = '85_e_s19_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 20, title = '85dB e vowel Subject 20 in frequency domain', filename = '85_e_s20_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 21, title = '85dB e vowel Subject 21 in frequency domain', filename = '85_e_s21_f_domain.png')
fig_mag_db(df_as_85_vsc_e, 22, title = '85dB e vowel Subject 22 in frequency domain', filename = '85_e_s22_f_domain.png')
'''

# plot 22 figures in one figure   
''' 
fig_mag_in_1(df_as_85_vsc_e, title = '85dB e vowel subjects in frequency domain', \
             path = path_result_freq, \
             filename = '85_e_f_domain.png')

fig_test_in_1(df_as_85_vsc_e, df_as_win_85_vsc_e, title = '85dB e vowel test in frequency domain', \
             path = path_result_freq, \
             filename = '85_e_f_domain_test.png')

fig_retest_in_1(df_as_85_vsc_e, df_as_win_85_vsc_e, title = '85dB e vowel retests in frequency domain', \
             path = path_result_freq, \
             filename = '85_e_f_domain_retest.png')
'''


# n vowel

'''
fig_mag_db(df_as_85_vsc_n, 1, title = '85dB n vowel Subject 1 in frequency domain', filename = '85_n_s1_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 2, title = '85dB n vowel Subject 2 in frequency domain', filename = '85_n_s2_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 3, title = '85dB n vowel Subject 3 in frequency domain', filename = '85_n_s3_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 4, title = '85dB n vowel Subject 4 in frequency domain', filename = '85_n_s4_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 5, title = '85dB n vowel Subject 5 in frequency domain', filename = '85_n_s5_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 6, title = '85dB n vowel Subject 6 in frequency domain', filename = '85_n_s6_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 7, title = '85dB n vowel Subject 7 in frequency domain', filename = '85_n_s7_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 8, title = '85dB n vowel Subject 8 in frequency domain', filename = '85_n_s8_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 9, title = '85dB n vowel Subject 9 in frequency domain', filename = '85_n_s9_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 10, title = '85dB n vowel Subject 10 in frequency domain', filename = '85_n_s10_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 11, title = '85dB n vowel Subject 11 in frequency domain', filename = '85_n_s11_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 12, title = '85dB n vowel Subject 12 in frequency domain', filename = '85_n_s12_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 13, title = '85dB n vowel Subject 13 in frequency domain', filename = '85_n_s13_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 14, title = '85dB n vowel Subject 14 in frequency domain', filename = '85_n_s14_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 15, title = '85dB n vowel Subject 15 in frequency domain', filename = '85_n_s15_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 16, title = '85dB n vowel Subject 16 in frequency domain', filename = '85_n_s16_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 17, title = '85dB n vowel Subject 17 in frequency domain', filename = '85_n_s17_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 18, title = '85dB n vowel Subject 18 in frequency domain', filename = '85_n_s18_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 19, title = '85dB n vowel Subject 19 in frequency domain', filename = '85_n_s19_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 20, title = '85dB n vowel Subject 20 in frequency domain', filename = '85_n_s20_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 21, title = '85dB n vowel Subject 21 in frequency domain', filename = '85_n_s21_f_domain.png')
fig_mag_db(df_as_85_vsc_n, 22, title = '85dB n vowel Subject 22 in frequency domain', filename = '85_n_s22_f_domain.png')
'''


# plot 22 figures in one figure    
'''
fig_mag_in_1(df_as_85_vsc_n, title = '85dB n vowel subjects in frequency domain', \
             path = path_result_freq, \
             filename = '85_n_f_domain.png')
fig_test_in_1(df_as_85_vsc_n, df_as_win_85_vsc_n, title = '85dB n vowel test in frequency domain', \
             path = path_result_freq, \
             filename = '85_n_f_domain_test.png')

fig_retest_in_1(df_as_85_vsc_n, df_as_win_85_vsc_n, title = '85dB n vowel retests in frequency domain', \
             path = path_result_freq, \
             filename = '85_n_f_domain_retest.png')
'''


# u vowel
'''
fig_mag_db(df_as_85_vsc_u, 1, title = '85dB u vowel Subject 1 in frequency domain', filename = '85_u_s1_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 2, title = '85dB u vowel Subject 2 in frequency domain', filename = '85_u_s2_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 3, title = '85dB u vowel Subject 3 in frequency domain', filename = '85_u_s3_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 4, title = '85dB u vowel Subject 4 in frequency domain', filename = '85_u_s4_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 5, title = '85dB u vowel Subject 5 in frequency domain', filename = '85_u_s5_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 6, title = '85dB u vowel Subject 6 in frequency domain', filename = '85_u_s6_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 7, title = '85dB u vowel Subject 7 in frequency domain', filename = '85_u_s7_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 8, title = '85dB u vowel Subject 8 in frequency domain', filename = '85_u_s8_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 9, title = '85dB u vowel Subject 9 in frequency domain', filename = '85_u_s9_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 10, title = '85dB u vowel Subject 10 in frequency domain', filename = '85_u_s10_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 11, title = '85dB u vowel Subject 11 in frequency domain', filename = '85_u_s11_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 12, title = '85dB u vowel Subject 12 in frequency domain', filename = '85_u_s12_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 13, title = '85dB u vowel Subject 13 in frequency domain', filename = '85_u_s13_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 14, title = '85dB u vowel Subject 14 in frequency domain', filename = '85_u_s14_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 15, title = '85dB u vowel Subject 15 in frequency domain', filename = '85_u_s15_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 16, title = '85dB u vowel Subject 16 in frequency domain', filename = '85_u_s16_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 17, title = '85dB u vowel Subject 17 in frequency domain', filename = '85_u_s17_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 18, title = '85dB u vowel Subject 18 in frequency domain', filename = '85_u_s18_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 19, title = '85dB u vowel Subject 19 in frequency domain', filename = '85_u_s19_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 20, title = '85dB u vowel Subject 20 in frequency domain', filename = '85_u_s20_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 21, title = '85dB u vowel Subject 21 in frequency domain', filename = '85_u_s21_f_domain.png')
fig_mag_db(df_as_85_vsc_u, 22, title = '85dB u vowel Subject 22 in frequency domain', filename = '85_u_s22_f_domain.png')
'''

# plot 22 figures in one figure   
''' 
fig_mag_in_1(df_as_85_vsc_u, title = '85dB u vowel subjects in frequency domain', \
             path = path_result_freq, \
             filename = '85_u_f_domain.png')

fig_test_in_1(df_as_85_vsc_u, df_as_win_85_vsc_u, title = '85dB u vowel test in frequency domain', \
             path = path_result_freq, \
             filename = '85_u_f_domain_test.png')

fig_retest_in_1(df_as_85_vsc_u, df_as_win_85_vsc_u, title = '85dB u vowel retests in frequency domain', \
             path = path_result_freq, \
             filename = '85_u_f_domain_retest.png')

'''


# example 
#correlation_matrix_01(corr_EFR_a_85_r_t_part, 'a_vowel_85Db cross correlation of test and retest')
'''
# EFR
correlation_matrix_01(corr_EFR_avg_85_a, 'cross correlation of 85dB a_vowel in time domain')
correlation_matrix_tt_01(corr_EFR_avg_85_a_t, 'cross correlation of 85dB a_vowel in time domain')
correlation_matrix_rr_01(corr_EFR_avg_85_a_re, 'cross correlation of 85dB a_vowel in time domain')

correlation_matrix_01(corr_EFR_avg_85_e, 'cross correlation of 85dB e_vowel in time domain')
correlation_matrix_tt_01(corr_EFR_avg_85_e_t, 'cross correlation of 85dB e_vowel in time domain')
correlation_matrix_rr_01(corr_EFR_avg_85_e_re, 'cross correlation of 85dB e_vowel in time domain')

correlation_matrix_01(corr_EFR_avg_85_n, 'cross correlation of 85dB n_vowel in time domain')
correlation_matrix_tt_01(corr_EFR_avg_85_n_t, 'cross correlation of 85dB n_vowel in time domain')
correlation_matrix_rr_01(corr_EFR_avg_85_n_re, 'cross correlation of 85dB n_vowel in time domain')

correlation_matrix_01(corr_EFR_avg_85_u, 'cross correlation of 85dB u_vowel in time domain')
correlation_matrix_tt_01(corr_EFR_avg_85_u_t, 'cross correlation of 85dB u_vowel in time domain')
correlation_matrix_rr_01(corr_EFR_avg_85_u_re, 'cross correlation of 85dB u_vowel in time domain')
'''

'''
# Amplitude Spectrum
correlation_matrix_01(corr_as_85_a, 'cross correlation of 85dB a_vowel in frequency domain')
correlation_matrix_01(corr_as_85_e, 'cross correlation of 85dB e_vowel in frequency domain')
correlation_matrix_01(corr_as_85_n, 'cross correlation of 85dB n_vowel in frequency domain')
correlation_matrix_01(corr_as_85_u, 'cross correlation of 85dB u_vowel in frequency domain')


# Amplitude Spectrum 7 points
correlation_matrix_01(corr_as7_85_a, 'cross correlation of 85dB a_vowel in frequency domain 7')
correlation_matrix_01(corr_as7_85_e, 'cross correlation of 85dB e_vowel in frequency domain 7')
correlation_matrix_01(corr_as7_85_n, 'cross correlation of 85dB n_vowel in frequency domain 7')
correlation_matrix_01(corr_as7_85_u, 'cross correlation of 85dB u_vowel in frequency domain 7')
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
"""
