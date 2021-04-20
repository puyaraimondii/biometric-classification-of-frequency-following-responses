#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:18:23 2019

@author: bruce
"""

import pandas as pd
import numpy as np
from scipy import fftpack
from scipy import signal
import matplotlib.pyplot as plt
import os
import librosa


'''
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
            f, temp_Cxy = signal.coherence(sig_in_1, sig_in_2, fs=9606, nperseg=256)
            
            # delete values lower than 0.01
            for l in range(len(temp_Cxy)):
                if temp_Cxy[l] < 0.1:
                    temp_Cxy[l] = 0
            # delete finish
            
            # test
            if i ==0 and j == 0:
                plt.figure()
                plt.semilogy(f, temp_Cxy)
                plt.title("test in complex_coherence_mx")
                plt.show()
            # test finish
            
            for k in range(len(temp_Cxy)):
                #test_t3 = (abs(temp_series[k]))**2
                #print(test_t3)
                temp_sum = temp_sum + abs(temp_Cxy[k])
            matrix_temp[i][j] = temp_sum
    output_3 = pd.DataFrame(matrix_temp)
    return output_3

'''


def concatenate_aenu(signal_input):
    # concatenate a e n u 4 vowels
    signal_concat_aenu = pd.DataFrame()
    for i in range(44):
        for j in range (2):
            signal_concat_aenu_temp1 = pd.DataFrame(signal_input.iloc[8*i+j, 0:1024].values.reshape(1, 1024))
            signal_concat_aenu_temp2 = pd.DataFrame(signal_input.iloc[8*i+j+2, 0:1024].values.reshape(1, 1024))
            signal_concat_aenu_temp3 = pd.DataFrame(signal_input.iloc[8*i+j+4, 0:1024].values.reshape(1, 1024))
            signal_concat_aenu_temp4 = pd.DataFrame(signal_input.iloc[8*i+j+6, 0:1024].values.reshape(1, 1024))
            signal_concat_aenu_label_temp = pd.DataFrame(signal_input.iloc[8*i+j, 1024:1031].values.reshape(1, 7))
            signal_concat_aenu_temp = pd.concat([signal_concat_aenu_temp1, signal_concat_aenu_temp2, signal_concat_aenu_temp3, signal_concat_aenu_temp4, signal_concat_aenu_label_temp], axis=1)
            signal_concat_aenu = signal_concat_aenu.append(signal_concat_aenu_temp, ignore_index = True)
            
    # set the title for df_EFR_85_aenu
    signal_concat_aenu.columns = np.append(np.arange(4096), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
    signal_concat_aenu_label = signal_concat_aenu.iloc[:, 4096:4103]
    
    return signal_concat_aenu, signal_concat_aenu_label


# swap an array
def swap(arr):  # only one dim
    l = len(arr)
    ret = np.zeros((l, 1))
    for i in range(l):
        ret[i] = arr[-i-1]
    return ret


def get_spectrum(wave_data, framerate, window_length_ms, window_shift_times):
    wav_length = len(wave_data)
    window_length = framerate * window_length_ms / 1000
    window_shift = int(window_length * window_shift_times)
    nframe = int(np.ceil((wav_length - window_length+1) / window_shift))
    freq_num = int(window_length/2)
    spec = np.zeros((freq_num, nframe))
    for i in range(nframe):
        start = int(i * window_shift)
        end = int(start + window_length)
        fft_slice = wave_data[start:end]
        fft_slice = fft_slice.astype('float64')
        fft_slice *= signal.hann(end-start, sym=False)
        w_fft = np.abs(np.fft.fft(fft_slice))
        w_fft = np.clip(w_fft, a_min=1e-16, a_max=None)
        freq = w_fft[:int(window_length/2)]
        freq = swap(freq).reshape(freq_num)
        spec[:, i] = freq
    return spec


def df_spectrum_txt(signal_input, store_path, store_name, plot_number=88, freq_end=4804):
    # example for input
    # signal_input = df_EFR_85_aenu_retest
    # store_path = '/home/bruce/Dropbox/Project/6.Result/data_spectrogram/'
    # store_name = 'EFR_85_retest'
    
    #subject_list = [01,02,03,04,06,07,08,09,11,12,13,14,15,16,17,18,19,20,21,22,23,25]
    subject_list = ['01','02','03','04','06','07','08','09','11','12','13','14','15','16','17','18','19','20','21','22','23','25']
    num_list = [1,2]
    
    for j in range(22):
        for k in range(2):
            signal_input_temp = signal_input.iloc[2*j+k, 0:4096]
            f, segment_times, spectrogram_output = signal.spectrogram(x=pd.to_numeric(signal_input_temp), fs=9606, nperseg=256, noverlap=128, nfft=9606)
            spectrogram_output = spectrogram_output[0:freq_end, :]
            
            # fs = 9606 nperseg=256, noverlap=128, nfft=1024 -> spectrogram.shape is 4804 * 31
            # pick 0 to 1300hz as 
            
            # plot for test
            if 2*j+k == plot_number:
                # test for plot
                plt.figure()
                plt.pcolormesh(segment_times, f, librosa.core.power_to_db(spectrogram_output), cmap='inferno')
                #plt.colorbar(cax, format='%+2.0f dB')
                plt.ylim(0, 1300) # 2048 -> 5000hz : 533 -> 1300
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.show()
            # plot for test finish
            
            np.savetxt(store_path + store_name+ '_%s_%s.txt' % (subject_list[j], num_list[k]), librosa.core.power_to_db(spectrogram_output))


def df_spectrum_txt_rename(signal_input, store_path, store_name, plot_number=88, freq_end=4804):
    # example for input
    # signal_input = df_EFR_85_aenu_retest
    # store_path = '/home/bruce/Dropbox/Project/6.Result/data_spectrogram/'
    # store_name = 'EFR_85_retest'
    
    #subject_list = [01,02,03,04,06,07,08,09,11,12,13,14,15,16,17,18,19,20,21,22,23,25]
    subject_list = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22']
    num_list = [1,2]
    
    for j in range(22):
        for k in range(2):
            signal_input_temp = signal_input.iloc[2*j+k, 0:4096]
            f, segment_times, spectrogram_output = signal.spectrogram(x=pd.to_numeric(signal_input_temp), fs=9606, nperseg=1024, noverlap=0, nfft=9606)
            spectrogram_output = spectrogram_output[0:freq_end, :]
            # fs = 9606 nperseg=256, noverlap=128, nfft=1024 -> spectrogram.shape is 4804 * 31
            # pick 0 to 1300hz as 
            
            # plot for test
            if 2*j+k == plot_number:
                # test for plot
                plt.figure()
                plt.pcolormesh(segment_times, f, librosa.core.power_to_db(spectrogram_output), cmap='inferno')
                #plt.colorbar(cax, format='%+2.0f dB')
                plt.ylim(0, 1300) # 2048 -> 5000hz : 533 -> 1300
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.show()
            # plot for test finish
            
            np.savetxt(store_path + store_name+ '_%s_%s.txt' % (subject_list[j], num_list[k]), librosa.core.power_to_db(spectrogram_output))


def df_spectrum_avg_txt_rename(signal_input, store_path, store_name, plot_number=88, freq_end=4804):
    # example for input
    # signal_input = df_EFR_85_aenu_retest
    # store_path = '/home/bruce/Dropbox/Project/6.Result/data_spectrogram/'
    # store_name = 'EFR_85_retest'
    
    #subject_list = [01,02,03,04,06,07,08,09,11,12,13,14,15,16,17,18,19,20,21,22,23,25]
    subject_list = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22']
    
    for j in range(22):
        signal_input_temp = signal_input.iloc[j, 0:4096]
        f, segment_times, spectrogram_output = signal.spectrogram(x=pd.to_numeric(signal_input_temp), fs=9606, nperseg=1024, noverlap=0, nfft=9606)
        spectrogram_output = spectrogram_output[0:freq_end, :]
        # fs = 9606 nperseg=256, noverlap=128, nfft=1024 -> spectrogram.shape is 4804 * 31
        # pick 0 to 1300hz as 
        
        # plot for test
        if j == plot_number:
            # test for plot
            plt.figure()
            plt.pcolormesh(segment_times, f, librosa.core.power_to_db(spectrogram_output), cmap='inferno')
            #plt.colorbar(cax, format='%+2.0f dB')
            plt.ylim(0, 1300) # 2048 -> 5000hz : 533 -> 1300
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()
        # plot for test finish
        
        np.savetxt(store_path + store_name+ '_%s.txt' % (subject_list[j]), librosa.core.power_to_db(spectrogram_output))


def df_spectrum_txt_peak_rename(signal_input, store_path, store_name, plot_number=88, freq_end=4804, peak_num=7):
    # example for input
    # signal_input = df_EFR_85_aenu_retest
    # store_path = '/home/bruce/Dropbox/Project/6.Result/data_spectrogram/'
    # store_name = 'EFR_85_retest'
    
    #subject_list = [01,02,03,04,06,07,08,09,11,12,13,14,15,16,17,18,19,20,21,22,23,25]
    subject_list = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22']
    num_list = [1,2]
    
    for j in range(22):
        for k in range(2):
            signal_input_temp = signal_input.iloc[2*j+k, 0:4096]
            f, segment_times, spectrogram_output = signal.spectrogram(x=pd.to_numeric(signal_input_temp), fs=9606, nperseg=256, noverlap=128, nfft=9606)
            spectrogram_output = spectrogram_output[0:freq_end, :]
            
            Sxx_peak = spectrogram_output[95:105, :]
            for i in range (peak_num):
                Sxx_temp = spectrogram_output[100*i+95:100*i+105, :]
                if i > 0:
                    Sxx_peak = np.concatenate((Sxx_peak, Sxx_temp), axis=0)
            Sxx_peak = Sxx_peak * 10e5
            # fs = 9606 nperseg=256, noverlap=128, nfft=1024 -> spectrogram.shape is 4804 * 31
            # pick 0 to 1300hz as 
            
            # plot for test
            if 2*j+k == plot_number:
                # test for plot
                plt.figure()
                plt.pcolormesh(segment_times, f, librosa.core.power_to_db(Sxx_peak), cmap='inferno')
                #plt.colorbar(cax, format='%+2.0f dB')
                plt.ylim(0, 1300) # 2048 -> 5000hz : 533 -> 1300
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                plt.show()
            # plot for test finish
            if (j==21 and k==1):
                print (Sxx_peak.shape)
                
            np.savetxt(store_path + store_name+ '_%s_%s.txt' % (subject_list[j], num_list[k]), librosa.core.power_to_db(Sxx_peak))

def df_time_txt_rename(signal_input, store_path, store_name):
    # example for input
    # signal_input = df_EFR_85_aenu_retest
    # store_path = '/home/bruce/Dropbox/Project/6.Result/data_spectrogram/'
    # store_name = 'EFR_85_retest'
    
    #subject_list = [01,02,03,04,06,07,08,09,11,12,13,14,15,16,17,18,19,20,21,22,23,25]
    subject_list = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22']
    num_list = [1,2]
    
    for j in range(22):
        for k in range(2):
            np.savetxt(store_path + store_name+ '_%s_%s.txt' % (subject_list[j], num_list[k]), signal_input.iloc[2*j+k, 0:4096])


#################################

f_dB = lambda x : 20 * np.log10(np.abs(x))

# import the pkl file
#pkl_file=pd.read_pickle('/Users/bruce/Documents/uOttawa/Project/audio_brainstem_response/Data_BruceSunMaster_Studies/study2/study2DataFrame.pkl')

# path for linux
df_EFR_input=pd.read_pickle('/home/bruce/Dropbox/Project/4.Code for Linux/df_EFR.pkl')
# path for mac
#df_EFR_input=pd.read_pickle('/Users/bruce/Dropbox/Project/4.Code for Linux/df_EFR.pkl')


# remove DC offset
df_EFR_detrend = pd.DataFrame()
for i in range(1408):
    # combine next two rows later
    df_EFR_detrend_temp = pd.DataFrame(signal.detrend(df_EFR_input.iloc[i: i+1, 0:1024], type='constant').reshape(1,1024))
    df_EFR_label = pd.DataFrame(df_EFR_input.iloc[i, 1024:1031].values.reshape(1,7))
    df_EFR_detrend = df_EFR_detrend.append(pd.concat([df_EFR_detrend_temp, df_EFR_label], axis=1, ignore_index=True))

# set the title of columns
df_EFR_detrend.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_EFR_detrend = df_EFR_detrend.reset_index(drop=True)

# df_EFR_label = pd.DataFrame(df_EFR.iloc[, 1024:1031].values.reshape(1,7))
df_EFR_detrend_label = df_EFR_input.iloc[:, 1024:1031]


# Time domain

# Define window function
win_kaiser = signal.kaiser(1024, beta=14)
win_hamming = signal.hamming(1024)


# implement the window function
df_EFR_win = pd.DataFrame()
for i in range(1408):
    # implement the window function
    df_EFR_t_win_temp = pd.DataFrame((df_EFR_detrend.iloc[i, 0:1024] * win_hamming).values.reshape(1,1024))
    df_EFR_label_temp = pd.DataFrame(df_EFR_detrend.iloc[i, 1024:1031].values.reshape(1,7))
    df_EFR_win = df_EFR_win.append(pd.concat([df_EFR_t_win_temp, df_EFR_label_temp], axis=1, ignore_index=True))
    
# set the title of columns
df_EFR = df_EFR_detrend.sort_values(by=["Condition", "Subject"])
df_EFR = df_EFR.reset_index(drop=True)
df_EFR_win.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_EFR_win = df_EFR_win.sort_values(by=["Condition", "Subject"])
df_EFR_win = df_EFR_win.reset_index(drop=True)


# filter by 'sound level, subjecta vowel and 85Db'
df_EFR_sorted = df_EFR.sort_values(by=["Sound Level","Condition", "Subject", "Vowel"])
df_EFR_sorted = df_EFR_sorted.reset_index(drop=True)
df_EFR_win_sorted = df_EFR_win.sort_values(by=["Sound Level","Condition", "Subject", "Vowel"])
df_EFR_win_sorted = df_EFR_win_sorted.reset_index(drop=True)


# zero padding

# for df_EFR
df_EFR_sorted_data = df_EFR_sorted.iloc[:, :1024]
df_EFR_sorted_label = df_EFR_sorted.iloc[:, 1024:]
df_EFR_sorted_mid = pd.DataFrame(np.zeros((1408, 95036)))
df_EFR_sorted_withzero = pd.concat([df_EFR_sorted_data, df_EFR_sorted_mid, df_EFR_sorted_label], axis=1)
# rename columns
df_EFR_sorted_withzero.columns = np.append(np.arange(96060), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])

# for df_EFR_win
df_EFR_win_sorted_data = df_EFR_win_sorted.iloc[:, :1024]
df_EFR_win_sorted_label = df_EFR_win_sorted.iloc[:, 1024:]
df_EFR_win_sorted_mid = pd.DataFrame(np.zeros((1408, 95036)))
df_EFR_win_sorted_withzero = pd.concat([df_EFR_win_sorted_data, df_EFR_win_sorted_mid, df_EFR_win_sorted_label], axis=1)
# rename columns
df_EFR_win_sorted_withzero.columns = np.append(np.arange(96060), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])


# separate based on sound levels

df_EFR_55 = pd.DataFrame(df_EFR_sorted.iloc[0:352, :])
df_EFR_55 = df_EFR_55.reset_index(drop=True)
df_EFR_65 = pd.DataFrame(df_EFR_sorted.iloc[352:704, :])
df_EFR_65 = df_EFR_65.reset_index(drop=True)
df_EFR_75 = pd.DataFrame(df_EFR_sorted.iloc[704:1056, :])
df_EFR_75 = df_EFR_75.reset_index(drop=True)
df_EFR_85 = pd.DataFrame(df_EFR_sorted.iloc[1056:1408, :])
df_EFR_85 = df_EFR_85.reset_index(drop=True)

df_EFR_win_55 = pd.DataFrame(df_EFR_win_sorted.iloc[0:352, :])
df_EFR_win_55 = df_EFR_win_55.reset_index(drop=True)
df_EFR_win_65 = pd.DataFrame(df_EFR_win_sorted.iloc[352:704, :])
df_EFR_win_65 = df_EFR_win_65.reset_index(drop=True)
df_EFR_win_75 = pd.DataFrame(df_EFR_win_sorted.iloc[704:1056, :])
df_EFR_win_75 = df_EFR_win_75.reset_index(drop=True)
df_EFR_win_85 = pd.DataFrame(df_EFR_win_sorted.iloc[1056:1408, :])
df_EFR_win_85 = df_EFR_win_85.reset_index(drop=True)
'''
df_EFR_0_55 = pd.DataFrame(df_EFR_sorted_withzero.iloc[0:352, :])
df_EFR_0_55 = df_EFR_0_55.reset_index(drop=True)
df_EFR_0_65 = pd.DataFrame(df_EFR_sorted_withzero.iloc[352:704, :])
df_EFR_0_65 = df_EFR_0_65.reset_index(drop=True)
df_EFR_0_75 = pd.DataFrame(df_EFR_sorted_withzero.iloc[704:1056, :])
df_EFR_0_75 = df_EFR_0_75.reset_index(drop=True)
df_EFR_0_85 = pd.DataFrame(df_EFR_sorted_withzero.iloc[1056:1408, :])
df_EFR_0_85 = df_EFR_0_85.reset_index(drop=True)

df_EFR_win_0_55 = pd.DataFrame(df_EFR_win_sorted_withzero.iloc[0:352, :])
df_EFR_win_0_55 = df_EFR_win_0_55.reset_index(drop=True)
df_EFR_win_0_65 = pd.DataFrame(df_EFR_win_sorted_withzero.iloc[352:704, :])
df_EFR_win_0_65 = df_EFR_win_0_65.reset_index(drop=True)
df_EFR_win_0_75 = pd.DataFrame(df_EFR_win_sorted_withzero.iloc[704:1056, :])
df_EFR_win_0_75 = df_EFR_win_0_75.reset_index(drop=True)
df_EFR_win_0_85 = pd.DataFrame(df_EFR_win_sorted_withzero.iloc[1056:1408, :])
df_EFR_win_0_85 = df_EFR_win_0_85.reset_index(drop=True)
'''

# test

    

# concatenate a e n u 4 vowels
'''
signal_concat = df_EFR_85
df_EFR_85_aenu = pd.DataFrame()
for i in range(44):
    for j in range (2):
        df_EFR_85_aenu_temp1 = pd.DataFrame(signal_concat.iloc[8*i+j, 0:1024].values.reshape(1, 1024))
        df_EFR_85_aenu_temp2 = pd.DataFrame(signal_concat.iloc[8*i+j+2, 0:1024].values.reshape(1, 1024))
        df_EFR_85_aenu_temp3 = pd.DataFrame(signal_concat.iloc[8*i+j+4, 0:1024].values.reshape(1, 1024))
        df_EFR_85_aenu_temp4 = pd.DataFrame(signal_concat.iloc[8*i+j+6, 0:1024].values.reshape(1, 1024))
        df_EFR_85_aenu_label_temp = pd.DataFrame(signal_concat.iloc[8*i+j, 1024:1031].values.reshape(1, 7))
        df_EFR_85_aenu_temp = pd.concat([df_EFR_85_aenu_temp1, df_EFR_85_aenu_temp2, df_EFR_85_aenu_temp3, df_EFR_85_aenu_temp4, df_EFR_85_aenu_label_temp], axis=1)
        df_EFR_85_aenu = df_EFR_85_aenu.append(df_EFR_85_aenu_temp, ignore_index = True)
        
# set the title for df_EFR_85_aenu
df_EFR_85_aenu.columns = np.append(np.arange(4096), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_EFR_85_aenu_label = df_EFR_85_aenu.iloc[:, 4096:4103]
'''

df_EFR_85_aenu, df_EFR_85_aenu_label = concatenate_aenu(df_EFR_85)

df_EFR_85_aenu_avg = pd.DataFrame()
for i in range(44):
    df_EFR_85_aenu_t = pd.DataFrame(df_EFR_85_aenu.iloc[2*i:2*i+2, 0:4096].mean(axis=0).values.reshape(1,4096))
    df_EFR_85_aenu_label = pd.DataFrame(df_EFR_85_aenu.iloc[2*i:2*i+1, 4096:].values.reshape(1,7))
    df_EFR_85_aenu_avg = df_EFR_85_aenu_avg.append(pd.concat([df_EFR_85_aenu_t, df_EFR_85_aenu_label], axis=1, ignore_index=True))
    
    

df_EFR_85_aenu_retest = df_EFR_85_aenu.iloc[0:44, :]
df_EFR_85_aenu_test = df_EFR_85_aenu.iloc[44:88, :]

df_EFR_85_aenu_avg_retest = df_EFR_85_aenu_avg.iloc[0:22, :].reset_index(drop=True)
df_EFR_85_aenu_avg_test = df_EFR_85_aenu_avg.iloc[22:44, :].reset_index(drop=True)
df_EFR_85_aenu_avg_retest_label = df_EFR_85_aenu_avg_retest.iloc[:, 4096:]

# test for spectrogram
f_1, t_1, Sxx_1 = signal.spectrogram(x=pd.to_numeric(df_EFR_85_aenu_retest.iloc[0, 0:4096]), fs=9606, nperseg=256, noverlap=128, nfft=9606)


print (Sxx_1.shape)
Sxx_2 = Sxx_1[0:1300, :]
print (Sxx_1.shape)


Sxx_peak = Sxx_2[85:115, :]
for i in range (7):
    print (i)
    Sxx_temp = Sxx_2[100*i+85:100*i+115, :]
    if i > 0:
        Sxx_peak = np.concatenate((Sxx_peak, Sxx_temp), axis=0)


plt.figure(1)
plt.pcolormesh(t_1, f_1, librosa.core.power_to_db(Sxx_1), cmap='inferno')
#plt.colorbar(cax, format='%+2.0f dB')
plt.ylim(0, 1300) # 2048 -> 5000hz : 533 -> 1300
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# test finished




# df_spectrogram
# spectrum fs=9606, nperseg=256, noverlap=128, nfft=9606

'''
df_spectrum_txt(signal_input=df_EFR_85_aenu_retest, 
                store_path='/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85/', 
                store_name='EFR_85_r')
df_spectrum_txt(signal_input=df_EFR_85_aenu_test, 
                store_path='/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85/', 
                store_name='EFR_85_t')


# shape [1300,31]
df_spectrum_txt(signal_input=df_EFR_85_aenu_retest, 
                store_path='/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_1300/', 
                store_name='EFR_85_r',
                freq_end=1300)
df_spectrum_txt(signal_input=df_EFR_85_aenu_test, 
                store_path='/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_1300/', 
                store_name='EFR_85_t',
                freq_end=1300)



# shape [700, 31]
df_spectrum_txt(signal_input=df_EFR_85_aenu_retest, 
                store_path='/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_800/', 
                store_name='EFR_85_r',
                freq_end=800)
df_spectrum_txt(signal_input=df_EFR_85_aenu_test, 
                store_path='/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_800/', 
                store_name='EFR_85_t',
                freq_end=800)



# shape [800, 31]
# parameter
# fs=9606, nperseg=256, noverlap=128, nfft=9606

df_spectrum_txt_rename(signal_input=df_EFR_85_aenu_retest, 
                       store_path='/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_800_rename/', 
                       store_name='EFR_85_r',
                       freq_end=800)
df_spectrum_txt_rename(signal_input=df_EFR_85_aenu_test, 
                       store_path='/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_800_rename/', 
                       store_name='EFR_85_t',
                       freq_end=800)

# shape [70, 31]
df_spectrum_txt_peak_rename(signal_input=df_EFR_85_aenu_retest, 
                            store_path='/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_peak_10_7_rename/', 
                            store_name='EFR_85_r',
                            freq_end=800,
                            peak_num=7)
df_spectrum_txt_peak_rename(signal_input=df_EFR_85_aenu_test, 
                            store_path='/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_peak_10_7_rename/', 
                            store_name='EFR_85_t',
                            freq_end=800,
                            peak_num=7)
'''


'''
# shape [800, 4]
df_spectrum_txt_rename(signal_input=df_EFR_85_aenu_retest, 
                       store_path='/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_800_4_rename/', 
                       store_name='EFR_85_r',
                       freq_end=800)
df_spectrum_txt_rename(signal_input=df_EFR_85_aenu_test, 
                       store_path='/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_800_4_rename/', 
                       store_name='EFR_85_t',
                       freq_end=800)
'''



'''
df_spectrum_avg_txt_rename(signal_input=df_EFR_85_aenu_avg_retest, 
                       store_path='/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_avg_800_4_rename/', 
                       store_name='EFR_85_r',
                       freq_end=800)
df_spectrum_avg_txt_rename(signal_input=df_EFR_85_aenu_avg_test, 
                       store_path='/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_avg_800_4_rename/', 
                       store_name='EFR_85_t',
                       freq_end=800)
'''

df_time_txt_rename(signal_input=df_EFR_85_aenu_retest, 
                       store_path='/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_time_4096_rename/retest/', 
                       store_name='EFR_85_r')
df_time_txt_rename(signal_input=df_EFR_85_aenu_test, 
                       store_path='/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_time_4096_rename/test/', 
                       store_name='EFR_85_t')