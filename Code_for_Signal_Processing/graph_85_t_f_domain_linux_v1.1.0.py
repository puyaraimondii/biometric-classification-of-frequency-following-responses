#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 22:33:07 2018

@author: bruce
"""

# threshold 


import pandas as pd
import numpy as np
from scipy import fftpack
from scipy import signal
from scipy import stats
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
        x_label = np.arange(0, 1024, 1)
        plt.plot(x_label, signal_in.iloc[2*i, :1024], '-')
        plt.plot(x_label, signal_in.iloc[2*i+1, :1024], '-')
        plt.ylabel(sub_title[i])
        plt.legend(('Retest', 'Test'), loc='upper right', fontsize='xx-small')
    plt.suptitle(title) # add a centered title to the figure
    plt.show()
    
    
def fig_sameday_time_in_1(signal_in, title = 'title'):
    plt.figure()
    sub_title = ['1', '2', '3', '4', '6', '7', '8', '9', '11', '12',\
                 '13', '14', '15', '16', '17', '18', '19', '20', '21',\
                 '22', '23', '25']
    for i in range(22):
        plt.subplot(11,2,i+1)
        x_label = np.arange(0, 1024, 1)
        plt.plot(x_label, signal_in.iloc[2*i, :1024], '-')
        plt.plot(x_label, signal_in.iloc[2*i+1, :1024], '-')
        pcc, p_value = stats.pearsonr(signal_in.iloc[2*i, :1024], signal_in.iloc[2*i+1, :1024])
        if pcc > 0.1:
            plt.ylabel(sub_title[i] + "\n%.3f" % pcc)
            print (sub_title[i] + " %.3f" % pcc)
        else:
            plt.ylabel(sub_title[i] + "\n%.3f" % pcc, color='r')
        #plt.text(1000, -0.2, "PCC is %.3f" % pcc, ha='right', va='top')
        plt.legend(('no.1', 'no.2'), loc='upper right', fontsize='xx-small')
        # print ("subject number:" , sub_title[i] , " PCC:" , pcc)
    plt.suptitle(title) # add a centered title to the figure
    plt.show()


def fig_sameday_time_in_1_avg_8(signal_in, title = 'title'):
    plt.figure()
    sub_title = ['1', '2', '3', '4', '6', '7', '8', '9', '11', '12',\
                 '13', '14', '15', '16', '17', '18', '19', '20', '21',\
                 '22', '23', '25']
    
    for i in range(22):
        pcc_sum=[]
        for j in range(8):
            pcc, p_value = stats.pearsonr(signal_in.iloc[44*j+2*i, :1024], signal_in.iloc[44*j+2*i+1, :1024])
            pcc_sum.append(pcc)
        pcc_avg = sum(pcc_sum)/len(pcc_sum)
        
        plt.subplot(11,2,i+1)
        x_label = np.arange(0, 1024, 1)
        plt.plot(x_label, signal_in.iloc[2*i, :1024], '-')
        plt.plot(x_label, signal_in.iloc[2*i+1, :1024], '-')
        
        if pcc_avg > 0.58:
            plt.ylabel(sub_title[i] + "\n%.3f" % pcc_avg)
            print (sub_title[i] + " %.3f" % pcc_avg)
        else:
            plt.ylabel(sub_title[i] + "\n%.3f" % pcc_avg, color='r')
        #plt.text(1000, -0.2, "PCC is %.3f" % pcc, ha='right', va='top')
        plt.legend(('no.1', 'no.2'), loc='upper right', fontsize='xx-small')
        # print ("subject number:" , sub_title[i] , " PCC:" , pcc)
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
    plt.suptitle(title) # add a centered title to the figure
    plt.show()
    
def fig_sameday_mag_in_1(signal_in, title = 'title'):
    plt.figure()
    sub_title = ['1', '2', '3', '4', '6', '7', '8', '9', '11', '12',\
                 '13', '14', '15', '16', '17', '18', '19', '20', '21',\
                 '22', '23', '25']
    for i in range(22):
        plt.subplot(11,2,i+1)
        x_label = np.arange(0, 4803, 0.1)
        plt.plot(x_label, signal_in.iloc[2*i, :48030], '-')
        plt.plot(x_label, signal_in.iloc[2*i+1, :48030], '-')
        pcc, p_value = stats.pearsonr(signal_in.iloc[2*i, :13000], signal_in.iloc[2*i+1, :13000])
        if pcc >0.95:
            plt.ylabel(sub_title[i]+ "\n%.3f" % pcc)
            print (sub_title[i]+ " %.3f" % pcc)
        else:
            plt.ylabel(sub_title[i]+ "\n%.3f" % pcc, color='r')
        plt.xlim(0,1300)
        # plt.ylim(top=100)
        plt.legend(('no.1', 'no.2'), loc='upper right', fontsize='xx-small')
        
    plt.suptitle(title) # add a centered title to the figure
    plt.show()


def fig_sameday_mag_in_1_avg_8(signal_in, title = 'title'):
    plt.figure()
    sub_title = ['1', '2', '3', '4', '6', '7', '8', '9', '11', '12',\
                 '13', '14', '15', '16', '17', '18', '19', '20', '21',\
                 '22', '23', '25']
    for i in range(22):
        pcc_sum=[]
        for j in range(8):
            pcc, p_value = stats.pearsonr(signal_in.iloc[44*j+2*i, :13000], signal_in.iloc[44*j+2*i+1, :13000])
            pcc_sum.append(pcc)
        pcc_avg = sum(pcc_sum)/len(pcc_sum)

        plt.subplot(11,2,i+1)
        x_label = np.arange(0, 4803, 0.1)
        plt.plot(x_label, signal_in.iloc[2*i, :48030], '-')
        plt.plot(x_label, signal_in.iloc[2*i+1, :48030], '-')
        
        if pcc_avg >0.875:
            plt.ylabel(sub_title[i]+ "\n%.3f" % pcc_avg)
            print (sub_title[i]+ " %.3f" % pcc_avg)
        else:
            plt.ylabel(sub_title[i]+ "\n%.3f" % pcc_avg, color='r')
        plt.xlim(0,1300)
        # plt.ylim(top=100)
        plt.legend(('no.1', 'no.2'), loc='upper right', fontsize='xx-small')
        
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
df_EFR=pd.read_pickle('/home/bruce/Dropbox/Project/4.Code for Linux/df_EFR.pkl')
# for mac
# df_EFR=pd.read_pickle('/Users/bruce/Dropbox/Project/4.Code for Linux/df_EFR.pkl')


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
df_EFR_85_mid = pd.DataFrame(np.zeros((352, 95036)))

df_EFR_win_85_data = df_EFR_win_85.iloc[:, :1024]
df_EFR_win_85_label = df_EFR_85_label
df_EFR_win_85_mid = df_EFR_85_mid

df_EFR_85_withzero = pd.concat([df_EFR_85_data, df_EFR_85_mid, df_EFR_85_label], axis=1)
df_EFR_85_withzero.columns = np.append(np.arange(96060), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])

df_EFR_win_85_withzero = pd.concat([df_EFR_win_85_data, df_EFR_85_mid, df_EFR_85_label], axis=1)
df_EFR_win_85_withzero.columns = np.append(np.arange(96060), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])


# normalization

# normalize the dataframe by standard deviation
df_EFR_85_data_std = df_EFR_85_data.std(axis=1)
df_EFR_85_data_norm_std = (df_EFR_85_data.iloc[:, :1024]).div(df_EFR_85_data_std, axis=0)


df_EFR_85_vsc = pd.concat([df_EFR_85_data, df_EFR_85_label], axis=1).sort_values(by=["Vowel", "Subject", "Condition"]).reset_index(drop=True)
df_EFR_85_vsc_norm_std = pd.concat([df_EFR_85_data_norm_std, df_EFR_85_label], axis=1).sort_values(by=["Vowel", "Subject", "Condition"]).reset_index(drop=True)
df_EFR_85_cvs = pd.concat([df_EFR_85_data, df_EFR_85_label], axis=1).sort_values(by=["Condition", "Vowel", "Subject"]).reset_index(drop=True)
df_EFR_85_withzero_vsc = df_EFR_85_withzero.sort_values(by=["Vowel", "Subject", "Condition"]).reset_index(drop=True)
df_EFR_85_withzero_cvs = df_EFR_85_withzero.sort_values(by=["Condition", "Vowel", "Subject"]).reset_index(drop=True)

df_EFR_85_cvs_r = df_EFR_85_cvs.iloc[0:176, :].reset_index(drop=True)
df_EFR_85_cvs_t = df_EFR_85_cvs.iloc[176:352, :].reset_index(drop=True)


# calculate the avarage PCC
pcc_sum=[]
for i in range (176):
    pcc_temp, p_value = stats.pearsonr(df_EFR_85_cvs.iloc[2*i, :1024], df_EFR_85_cvs.iloc[2*i+1, :1024])
    pcc_sum.append(pcc_temp)
pcc_avg = sum(pcc_sum)/len(pcc_sum)


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


df_EFR_85_cvs_r_a = df_EFR_85_cvs_r.iloc[0:44, :].reset_index(drop=True)
df_EFR_85_cvs_r_a_label = df_EFR_85_cvs_r.iloc[0:44, 1024:]
df_EFR_85_cvs_r_e = df_EFR_85_cvs_r.iloc[44:88, :].reset_index(drop=True)
df_EFR_85_cvs_r_n = df_EFR_85_cvs_r.iloc[88:132, :].reset_index(drop=True)
df_EFR_85_cvs_r_u = df_EFR_85_cvs_r.iloc[132:176, :].reset_index(drop=True)

df_EFR_85_cvs_t_a = df_EFR_85_cvs_t.iloc[0:44, :].reset_index(drop=True)
df_EFR_85_cvs_t_e = df_EFR_85_cvs_t.iloc[44:88, :].reset_index(drop=True)
df_EFR_85_cvs_t_n = df_EFR_85_cvs_t.iloc[88:132, :].reset_index(drop=True)
df_EFR_85_cvs_t_u = df_EFR_85_cvs_t.iloc[132:176, :].reset_index(drop=True)


# frequency domain


# for df_EFR_avg_vcs_withzero
df_as_85_no0= pd.DataFrame()
df_as_85= pd.DataFrame()
df_as7_85= pd.DataFrame()
df_as_win_85= pd.DataFrame()
df_as7_win_85= pd.DataFrame()
for i in range(352):
    #temp_aenu_EFR = df_EFR_avg_aenu_withzero.iloc[i, 0:9606]    
    temp_as = np.abs((np.fft.fft(df_EFR_85_data.iloc[i, :], n=96060))[range(int(n2/2))])
    df_as_85 = df_as_85.append(pd.DataFrame(temp_as.reshape(1,48030)), ignore_index = True)
    df_as7_85 = df_as7_85.append(pd.DataFrame(np.array([temp_as[1000], temp_as[2000], temp_as[3000], temp_as[4000], \
                                                            temp_as[5000], temp_as[6000], temp_as[7000]]).reshape(1,7)), ignore_index = True)
    temp_as_win = np.abs((np.fft.fft(df_EFR_win_85_data.iloc[i, :], n=96060))[range(int(n2/2))])
    df_as_win_85 = df_as_win_85.append(pd.DataFrame(temp_as_win.reshape(1,48030)), ignore_index = True)
    df_as7_win_85 = df_as7_win_85.append(pd.DataFrame(np.array([temp_as_win[1000], temp_as_win[2000], temp_as_win[3000], temp_as_win[4000], \
                                                            temp_as_win[5000], temp_as_win[6000], temp_as_win[7000]]).reshape(1,7)), ignore_index = True)

df_as_85 = pd.concat([df_as_85, df_EFR_85_label], axis=1) # add labels on it
df_as7_85 = pd.concat([df_as7_85, df_EFR_85_label], axis=1) # add labels on it
df_as_win_85 = pd.concat([df_as_win_85, df_EFR_win_85_label], axis=1) # add labels on it
df_as7_win_85 = pd.concat([df_as7_win_85, df_EFR_win_85_label], axis=1) # add labels on it

#resort df_as_85 based on 1.vowel, 2.subject 3.condition
df_as_85_vsc = df_as_85.sort_values(by=["Vowel", "Subject", "Condition"]).reset_index(drop=True)
df_as_85_vsc_label = df_as_85_vsc.iloc[:,48030:]
df_as_win_85_vsc = df_as_win_85.sort_values(by=["Vowel", "Subject", "Condition"]).reset_index(drop=True)
df_as_win_85_vsc_label = df_as_win_85_vsc.iloc[:,48030:]

df_as_85_cvs = df_as_85.sort_values(by=["Condition", "Vowel", "Subject"]).reset_index(drop=True)
df_as_85_cvs_label = df_as_85_cvs.iloc[:,48030:]
df_as_win_85_cvs = df_as_win_85.sort_values(by=["Condition", "Vowel", "Subject"]).reset_index(drop=True)
df_as_win_85_cvs_label = df_as_win_85_cvs.iloc[:,48030:]


df_as_85_cvs_r = df_as_85_cvs.iloc[0:176, :].reset_index(drop=True)
df_as_85_cvs_t = df_as_85_cvs.iloc[176:352, :].reset_index(drop=True)

df_as_win_85_cvs_r = df_as_win_85_cvs.iloc[0:176, :].reset_index(drop=True)
df_as_win_85_cvs_t = df_as_win_85_cvs.iloc[176:352, :].reset_index(drop=True)

# calculate the avarage PCC
pcc_as_sum=[]
for i in range (176):
    pcc_as_temp, p_value = stats.pearsonr(df_as_win_85_cvs.iloc[2*i, :13000], df_as_win_85_cvs.iloc[2*i+1, :13000])
    pcc_as_sum.append(pcc_as_temp)
pcc_as_avg = sum(pcc_as_sum)/len(pcc_as_sum)


df_as_85_cvs_r_a = df_as_85_cvs_r.iloc[0:44, :].reset_index(drop=True)
df_as_85_cvs_r_e = df_as_85_cvs_r.iloc[44:88, :].reset_index(drop=True)
df_as_85_cvs_r_n = df_as_85_cvs_r.iloc[88:132, :].reset_index(drop=True)
df_as_85_cvs_r_u = df_as_85_cvs_r.iloc[132:176, :].reset_index(drop=True)

df_as_85_cvs_t_a = df_as_85_cvs_t.iloc[0:44, :].reset_index(drop=True)
df_as_85_cvs_t_e = df_as_85_cvs_t.iloc[44:88, :].reset_index(drop=True)
df_as_85_cvs_t_n = df_as_85_cvs_t.iloc[88:132, :].reset_index(drop=True)
df_as_85_cvs_t_u = df_as_85_cvs_t.iloc[132:176, :].reset_index(drop=True)


df_as_win_85_cvs_r_a = df_as_win_85_cvs_r.iloc[0:44, :].reset_index(drop=True)
df_as_win_85_cvs_r_e = df_as_win_85_cvs_r.iloc[44:88, :].reset_index(drop=True)
df_as_win_85_cvs_r_n = df_as_win_85_cvs_r.iloc[88:132, :].reset_index(drop=True)
df_as_win_85_cvs_r_u = df_as_win_85_cvs_r.iloc[132:176, :].reset_index(drop=True)

df_as_win_85_cvs_t_a = df_as_win_85_cvs_t.iloc[0:44, :].reset_index(drop=True)
df_as_win_85_cvs_t_e = df_as_win_85_cvs_t.iloc[44:88, :].reset_index(drop=True)
df_as_win_85_cvs_t_n = df_as_win_85_cvs_t.iloc[88:132, :].reset_index(drop=True)
df_as_win_85_cvs_t_u = df_as_win_85_cvs_t.iloc[132:176, :].reset_index(drop=True)

# plot

# plot the time domain signal


'''
fig_sameday_time_in_1(df_EFR_85_cvs_r_a, title= '85dB a vowel retest subjects in time domain')
fig_sameday_time_in_1(df_EFR_85_cvs_r_e, title= '85dB e vowel retest subjects in time domain')
fig_sameday_time_in_1(df_EFR_85_cvs_r_n, title= '85dB n vowel retest subjects in time domain')
fig_sameday_time_in_1(df_EFR_85_cvs_r_u, title= '85dB u vowel retest subjects in time domain')

fig_sameday_time_in_1(df_EFR_85_cvs_t_a, title= '85dB a vowel test subjects in time domain')
fig_sameday_time_in_1(df_EFR_85_cvs_t_e, title= '85dB e vowel test subjects in time domain')
fig_sameday_time_in_1(df_EFR_85_cvs_t_n, title= '85dB n vowel test subjects in time domain')
fig_sameday_time_in_1(df_EFR_85_cvs_t_u, title= '85dB u vowel test subjects in time domain')
'''

fig_sameday_time_in_1_avg_8(df_EFR_85_cvs, title= '85dB subjects in time domain')

'''
fig_sameday_mag_in_1(df_as_85_cvs_r_a, title = '85dB a vowel retest subjects in frequency domain')
fig_sameday_mag_in_1(df_as_85_cvs_r_e, title = '85dB e vowel retest subjects in frequency domain')
fig_sameday_mag_in_1(df_as_85_cvs_r_n, title = '85dB n vowel retest subjects in frequency domain')
fig_sameday_mag_in_1(df_as_85_cvs_r_u, title = '85dB u vowel retest subjects in frequency domain')

fig_sameday_mag_in_1(df_as_85_cvs_t_a, title = '85dB a vowel test subjects in frequency domain')
fig_sameday_mag_in_1(df_as_85_cvs_t_e, title = '85dB e vowel test subjects in frequency domain')
fig_sameday_mag_in_1(df_as_85_cvs_t_n, title = '85dB n vowel test subjects in frequency domain')
fig_sameday_mag_in_1(df_as_85_cvs_t_u, title = '85dB u vowel test subjects in frequency domain')
'''

'''
fig_sameday_mag_in_1(df_as_win_85_cvs_r_a, title = '85dB a vowel retest subjects in frequency domain(hamming)')
fig_sameday_mag_in_1(df_as_win_85_cvs_r_e, title = '85dB e vowel retest subjects in frequency domain(hamming)')
fig_sameday_mag_in_1(df_as_win_85_cvs_r_n, title = '85dB n vowel retest subjects in frequency domain(hamming)')
fig_sameday_mag_in_1(df_as_win_85_cvs_r_u, title = '85dB u vowel retest subjects in frequency domain(hamming)')

fig_sameday_mag_in_1(df_as_win_85_cvs_t_a, title = '85dB a vowel test subjects in frequency domain(hamming)')
fig_sameday_mag_in_1(df_as_win_85_cvs_t_e, title = '85dB e vowel test subjects in frequency domain(hamming)')
fig_sameday_mag_in_1(df_as_win_85_cvs_t_n, title = '85dB n vowel test subjects in frequency domain(hamming)')
fig_sameday_mag_in_1(df_as_win_85_cvs_t_u, title = '85dB u vowel test subjects in frequency domain(hamming)')
'''

fig_sameday_mag_in_1_avg_8(df_as_win_85_cvs, title = '85dB subjects in frequency domain(hamming)')