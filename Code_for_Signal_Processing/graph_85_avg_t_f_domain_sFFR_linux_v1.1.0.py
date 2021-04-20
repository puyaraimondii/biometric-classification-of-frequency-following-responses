#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:55:04 2019

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
    ylabels = ['T1','T2','T3','T4','T6','T7','T8','T9', 'T11', 'T12', 'T13', 'T14', 
               'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T25']
    xlabels = ['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 
               'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
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
    # cmap = cm.get_cmap('jet', 30)
    cax = ax1.matshow(output, cmap='binary')
    # cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    # fig.colorbar(cax)
    ax1.grid(False)
    plt.title(cm_title)
    ylabels=['T1','T2','T3','T4','T6','T7','T8','T9', 'T11', 'T12', 'T13', 'T14', 
             'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T25']
    xlabels=['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 
             'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
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
    ylabels=['T1','T2','T3','T4','T6','T7','T8','T9', 'T11', 'T12', 'T13', 'T14', 
             'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T25']
    xlabels=['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 
             'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
    ax1.set_xticks(np.arange(len(xlabels)))
    ax1.set_yticks(np.arange(len(ylabels)))
    ax1.set_xticklabels(xlabels,fontsize=6)
    ax1.set_yticklabels(ylabels,fontsize=6)
    
    # figure 2 
    im2 = ax2.matshow(output2, cmap='binary')
    #fig.colorbar(im2, ax2)
    ax2.grid(False)
    ax2.set_title(cm_title2)
    ylabels=['T1','T2','T3','T4','T6','T7','T8','T9', 'T11', 'T12', 'T13', 'T14', 
             'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T25']
    xlabels=['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 
             'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
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
    ylabels=['T1','T2','T3','T4','T6','T7','T8','T9', 'T11', 'T12', 'T13', 'T14', 
             'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T25']
    xlabels=['T1','T2','T3','T4','T6','T7','T8','T9', 'T11', 'T12', 'T13', 'T14', 
             'T15', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T22', 'T23', 'T25']
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
    ylabels=['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 
             'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
    xlabels=['R1','R2','R3','R4','R6','R7','R8','R9', 'R11', 'R12', 'R13', 'R14', 
             'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R25']
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


# plot time domain signal in one figure
def fig_time_in_1(signal_in, title = 'title'):
    plt.figure()
    sub_title = ['1', '2', '3', '4', '6', '7', '8', '9', '11', '12',\
                 '13', '14', '15', '16', '17', '18', '19', '20', '21',\
                 '22', '23', '25']
    for i in range(22):
        plt.subplot(11,2,i+1)
        x_label = np.arange(0, 100, 0.09765625)
        plt.plot(x_label, signal_in.iloc[2*i, :1024], '-')
        plt.plot(x_label, signal_in.iloc[2*i+1, :1024], '-')
        plt.ylabel(sub_title[i])
        plt.legend(('Retest', 'Test'), loc='upper right', fontsize='xx-small')
        if i < 20:
            plt.xticks([])
        else:
            plt.xlabel('Time (ms)')
    plt.suptitle(title) # add a centered title to the figure
    plt.show()


# plot frequency domain signal in one figure
def fig_mag_in_1(signal_in, title = 'title'):
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
        plt.xlim(0,1300)
        plt.legend(('Retest', 'Test'), loc='upper right', fontsize='xx-small')
        if i < 20:
            plt.xticks([])
        else:
            plt.xlabel('Frequency(Hz)')
    plt.suptitle(title) # add a centered title to the figure
    plt.show()
    


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
            f, temp_Cxy = signal.coherence(sig_in_1, sig_in_2, fs=9606, nperseg=96)
            
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


def fig_coherence_in_1(signal_in, threshold_Cxy = None, title = 'title', title2 = 'title2'):
    # threshold_Cxy is used for setting minimum value
    Cxy_sum = pd.DataFrame()
    plt.figure()
    sub_title = ['1', '2', '3', '4', '6', '7', '8', '9', '11', '12',\
                 '13', '14', '15', '16', '17', '18', '19', '20', '21',\
                 '22', '23', '25']
    for i in range(22):
        sig_in_1 = signal_in.iloc[i, :]
        sig_in_2 = signal_in.iloc[i+22, :]
        # signal 9606Hz length 106.6ms window length 10ms -> nperseg=96
        # no zero padding
        # f, temp_Cxy = signal.coherence(sig_in_1, sig_in_2, fs=9606, nperseg=128)
        # with zero padding
        f, temp_Cxy = signal.coherence(sig_in_1, sig_in_2, fs = 9606, nperseg=512, nfft=19210)
        
        # print("shape of temp_Cxy is")
        # print (temp_Cxy.shape)
        
        # delete value lower than 0.05
        if (threshold_Cxy != None):
            for l in range(len(temp_Cxy)):
                if temp_Cxy[l] < threshold_Cxy:
                    temp_Cxy[l] = 0
        # delete finish
        
        Cxy_sum = Cxy_sum.append(pd.DataFrame(np.reshape(temp_Cxy, (1,9606))), ignore_index=True)
        
        plt.subplot(11,2,i+1)
        plt.plot(f, temp_Cxy)
        plt.ylabel(sub_title[i])
        plt.xlim(0,2000)
        plt.legend(('Retest', 'Test'), loc='upper right', fontsize='xx-small')

    plt.suptitle(title) # add a centered title to the figure
    plt.show()


    
    # plot aveerage of 22 subjects
    plt.figure()
    plt.subplot(1,1,1)
    Cxy_avg = Cxy_sum.mean(axis=0)
    plt.plot(f, Cxy_avg)
    plt.title('average of 22 subjects based on '+ title2)
    plt.xlim(0,2000)
    plt.show()

#################################


f_dB = lambda x : 20 * np.log10(np.abs(x))


# import the pkl file

# for linux
df_FFR=pd.read_pickle('/home/bruce/Dropbox/Project/4.Code for Linux/df_FFR.pkl')
# for mac
# df_FFR=pd.read_pickle('/Users/bruce/Dropbox/Project/4.Code for Linux/df_FFR.pkl')

# remove DC offset
df_FFR_detrend = pd.DataFrame()
for i in range(1408):
    # combine next two rows later
    df_FFR_detrend_data_t = pd.DataFrame(signal.detrend(df_FFR.iloc[i: i+1, 0:1024], type='constant').reshape(1,1024))
    df_FFR_label_t = pd.DataFrame(df_FFR.iloc[i, 1024:1031].values.reshape(1,7))
    df_FFR_detrend = df_FFR_detrend.append(pd.concat([df_FFR_detrend_data_t, df_FFR_label_t], axis=1, ignore_index=True))

# set the title of columns
df_FFR_detrend.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_FFR_detrend = df_FFR_detrend.reset_index(drop=True)
df_FFR = df_FFR_detrend


# Time domain

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
    
    # implement the window function
    df_FFR_avg_t_win = pd.DataFrame((df_FFR_avg_t.iloc[0,:] * win_hamming).values.reshape(1,1024))
    df_FFR_label = pd.DataFrame(df_FFR.iloc[2*i, 1024:1031].values.reshape(1,7))
    df_FFR_avg = df_FFR_avg.append(pd.concat([df_FFR_avg_t, df_FFR_label], axis=1, ignore_index=True))
    df_FFR_avg_win = df_FFR_avg_win.append(pd.concat([df_FFR_avg_t_win, df_FFR_label], axis=1, ignore_index=True))
    
# set the title of columns
df_FFR_avg.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_FFR_avg = df_FFR_avg.sort_values(by=["Condition", "Subject"]).reset_index(drop=True)

df_FFR_avg_win.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_FFR_avg_win = df_FFR_avg_win.sort_values(by=["Condition", "Subject"]).reset_index(drop=True)



# average all the subjects , test and retest and keep one sound levels
# filter by 'a vowel and 85Db'
df_FFR_avg_sorted = df_FFR_avg.sort_values(by=["Sound Level", "Vowel","Condition", "Subject"]).reset_index(drop=True)
df_FFR_avg_win_sorted = df_FFR_avg_win.sort_values(by=["Sound Level", "Vowel","Condition", "Subject"]).reset_index(drop=True)


# filter55 65 75 sound levels and keep 85dB
# keep vowel condition and subject
df_FFR_avg_85 = pd.DataFrame(df_FFR_avg_sorted.iloc[528:, :])
df_FFR_avg_85 = df_FFR_avg_85.reset_index(drop=True)
df_FFR_avg_win_85 = pd.DataFrame(df_FFR_avg_win_sorted.iloc[528:, :])
df_FFR_avg_win_85 = df_FFR_avg_win_85.reset_index(drop=True)

 
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




# for df_FFR
df_FFR_data = df_FFR.iloc[:, :1024]
df_FFR_label = df_FFR.iloc[:, 1024:]

# for df_FFR_avg_85
df_FFR_avg_85_data = df_FFR_avg_85.iloc[:, :1024]
df_FFR_avg_85_label = df_FFR_avg_85.iloc[:, 1024:]

# normalization

# normalize the dataframe by standard deviation
df_FFR_avg_85_data_std = df_FFR_avg_85_data.std(axis=1)
df_FFR_avg_85_data_norm_std = (df_FFR_avg_85_data.iloc[:, :1024]).div(df_FFR_avg_85_data_std, axis=0)
# normalize the dataframe by 
df_FFR_avg_85_data_sumofsquare = (np.square(df_FFR_avg_85_data.iloc[:, :1024])).sum(axis=1)
df_FFR_avg_85_data_norm_ss = (df_FFR_avg_85_data.iloc[:, :1024]).div(df_FFR_avg_85_data_sumofsquare, axis=0)


df_FFR_avg_85_vsc = pd.concat([df_FFR_avg_85_data, df_FFR_avg_85_label], axis=1).sort_values(by=["Vowel", "Subject", "Condition"]).reset_index(drop=True)
df_FFR_avg_85_vsc_norm_std = pd.concat([df_FFR_avg_85_data_norm_std, df_FFR_avg_85_label], axis=1).sort_values(by=["Vowel", "Subject", "Condition"]).reset_index(drop=True)
df_FFR_avg_85_vsc_norm_ss = pd.concat([df_FFR_avg_85_data_norm_ss, df_FFR_avg_85_label], axis=1).sort_values(by=["Vowel", "Subject", "Condition"]).reset_index(drop=True)


df_FFR_avg_85_vsc_a = df_FFR_avg_85_vsc.iloc[0:44, :1024]
df_FFR_avg_85_vsc_e = df_FFR_avg_85_vsc.iloc[44:88, :1024]
df_FFR_avg_85_vsc_n = df_FFR_avg_85_vsc.iloc[88:132, :1024]
df_FFR_avg_85_vsc_u = df_FFR_avg_85_vsc.iloc[132:176, :1024]

df_FFR_avg_85_vsc_norm_std_a = df_FFR_avg_85_vsc_norm_std.iloc[0:44, :1024]
df_FFR_avg_85_vsc_norm_std_e = df_FFR_avg_85_vsc_norm_std.iloc[44:88, :1024]
df_FFR_avg_85_vsc_norm_std_n = df_FFR_avg_85_vsc_norm_std.iloc[88:132, :1024]
df_FFR_avg_85_vsc_norm_std_u = df_FFR_avg_85_vsc_norm_std.iloc[132:176, :1024]

df_FFR_avg_85_vsc_norm_ss_a = df_FFR_avg_85_vsc_norm_ss.iloc[0:44, :1024]
df_FFR_avg_85_vsc_norm_ss_e = df_FFR_avg_85_vsc_norm_ss.iloc[44:88, :1024]
df_FFR_avg_85_vsc_norm_ss_n = df_FFR_avg_85_vsc_norm_ss.iloc[88:132, :1024]
df_FFR_avg_85_vsc_norm_ss_u = df_FFR_avg_85_vsc_norm_ss.iloc[132:176, :1024]


# concatenate AENU
temp1 = pd.concat([df_FFR_avg_85_vsc.iloc[0:44, 0:1024].reset_index(drop=True),
                   df_FFR_avg_85_vsc.iloc[44:88, 0:1024].reset_index(drop=True)], axis=1)
temp2 = pd.concat([df_FFR_avg_85_vsc.iloc[88:132, 0:1024].reset_index(drop=True), 
                   df_FFR_avg_85_vsc.iloc[132:176, 0:1024].reset_index(drop=True)], axis=1)
df_FFR_avg_85_aenu = pd.concat([temp1, temp2], axis=1, ignore_index=True)


# df_FFR_avg_win_85
df_FFR_avg_win_85_data = df_FFR_avg_win_85.iloc[:, :1024]
df_FFR_avg_win_85_label = df_FFR_avg_win_85.iloc[:, 1024:]


df_FFR_win_85_as= pd.DataFrame()
df_FFR_win_85_as7 = pd.DataFrame()
for i in range(176):
    temp_as_win = 2/n2 *np.abs((np.fft.fft(df_FFR_avg_win_85_data.iloc[i, :], n=96060))[range(int(n2/2))])
    df_FFR_win_85_as = df_FFR_win_85_as.append(pd.DataFrame(temp_as_win.reshape(1,48030)), ignore_index = True)
    df_FFR_win_85_as7 = df_FFR_win_85_as7.append(pd.DataFrame(np.array([temp_as_win[1000], temp_as_win[2000], temp_as_win[3000], temp_as_win[4000], \
                                                            temp_as_win[5000], temp_as_win[6000], temp_as_win[7000]]).reshape(1,7)), ignore_index = True)


df_FFR_win_85_as = pd.concat([df_FFR_win_85_as, df_FFR_avg_win_85_label], axis=1) # add labels on it
df_FFR_win_85_as7 = pd.concat([df_FFR_win_85_as7, df_FFR_avg_win_85_label], axis=1) # add labels on it


df_FFR_win_85_as_vsc = df_FFR_win_85_as.sort_values(by=["Vowel", "Subject", "Condition"])
df_FFR_win_85_as_vsc = df_FFR_win_85_as_vsc.reset_index(drop=True)
df_FFR_win_85_as_vsc_label = df_FFR_win_85_as_vsc.iloc[:,48030:]

df_FFR_win_85_as_vsc_a = df_FFR_win_85_as_vsc.iloc[0:44, :]
df_FFR_win_85_as_vsc_e = df_FFR_win_85_as_vsc.iloc[44:88, :]
df_FFR_win_85_as_vsc_n = df_FFR_win_85_as_vsc.iloc[88:132, :]
df_FFR_win_85_as_vsc_u = df_FFR_win_85_as_vsc.iloc[132:176, :]



# plot

# plot the time domain signal

fig_time_in_1(df_FFR_avg_85_vsc_a, title= '85dB a vowel spectral FFRs in time domain')
fig_time_in_1(df_FFR_avg_85_vsc_e, title= '85dB e vowel spectral FFRs in time domain')
fig_time_in_1(df_FFR_avg_85_vsc_n, title= '85dB n vowel spectral FFRs in time domain')
fig_time_in_1(df_FFR_avg_85_vsc_u, title= '85dB u vowel spectral FFRs in time domain')

# plot the frequency domain signal

fig_mag_in_1(df_FFR_win_85_as_vsc_a, title = '85dB a vowel spectral FFRs in frequency domain')
fig_mag_in_1(df_FFR_win_85_as_vsc_e, title = '85dB e vowel spectral FFRs in frequency domain')
fig_mag_in_1(df_FFR_win_85_as_vsc_n, title = '85dB n vowel spectral FFRs in frequency domain')
fig_mag_in_1(df_FFR_win_85_as_vsc_u, title = '85dB u vowel spectral FFRs in frequency domain')
