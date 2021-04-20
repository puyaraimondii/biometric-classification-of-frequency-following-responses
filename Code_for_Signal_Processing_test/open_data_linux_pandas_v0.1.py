#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:04:07 2018

@author: bruce
"""

import pandas as pd
import os
import numpy as np
from scipy import fftpack
from scipy import signal
import matplotlib.pyplot as plt

pkl_file=pd.read_pickle('/home/bruce/Documents/uOttawa/Project/Data_BrianHeffernanPhDStudies/study2/study2DataFrame.pkl')

# parameters
sampling_rate = 9606
n = 1024
k = np.arange(n)
T = n/sampling_rate
frq = k/T
freq = frq[range(int(n/2))]

# initialize an array
title = np.hstack((np.arange(1024).reshape(1,1024) + np.array(("Subject", "Sex", "Condition", "Vowel", "Sound Level", "Number", "Avg_type")).reshape(1,7))

# set a dataframe
df = pd.DataFrame(data, columns = title)


# import s1_test_a_85dB
root_dir = "/home/bruce/Documents/uOttawa/Project/Data_BrianHeffernanPhDStudies/study2/rawdata/male/s1/test/a vowel/85"
s1_m_t_a_85_1_EFR = np.array([])
s1_m_t_a_85_2_EFR = np.array([])
s1_m_t_a_85_avg_EFR = np.array([])
# Number 1
f = open(root_dir+"/"+"1.txt", 'r')
data = f.readlines()
for line in data:
    #Condensation, Rarefaction, EFR, FFR = line.split(",")
    _, _, temp_EFR, _ = line.split(",")
    s1_m_t_a_85_1_EFR = np.hstack((s1_m_t_a_85_1_EFR, temp_EFR))
s1_m_t_a_85_1_EFR = s1_m_t_a_85_1_EFR[1:]
s1_m_t_a_85_1_EFR = s1_m_t_a_85_1_EFR.reshape(1,1024)
# Number 2
f = open(root_dir+"/"+"2.txt", 'r')
data = f.readlines()
for line in data:
    _, _, temp_EFR, _ = line.split(",")
    s1_m_t_a_85_2_EFR = np.hstack((s1_m_t_a_85_2_EFR, temp_EFR))
s1_m_t_a_85_2_EFR = s1_m_t_a_85_2_EFR[1:]
s1_m_t_a_85_2_EFR = s1_m_t_a_85_2_EFR.reshape(1,1024)
# calculate the avg
for i in range(0,n):
    s1_m_t_a_85_avg_EFR = np.append(s1_m_t_a_85_avg_EFR, (np.float64(s1_m_t_a_85_1_EFR[i]) + np.float64(s1_m_t_a_85_2_EFR[i]))/2)
s1_m_t_a_85_avg_EFR = s1_m_t_a_85_avg_EFR.reshape(1,1024)
#s1_m_t_a_85_avg_EFR.append(["s1", "m", "test", "a vowel", "85", "1", "EFR"])

# Calculate the spectrum
s1_m_t_a_85_1_EFR_amplitude_spectrum = np.abs((fftpack.fft(s1_m_t_a_85_1_EFR)/n)[range(int(n/2))])
s1_m_t_a_85_2_EFR_amplitude_spectrum = np.abs((fftpack.fft(s1_m_t_a_85_2_EFR)/n)[range(int(n/2))])