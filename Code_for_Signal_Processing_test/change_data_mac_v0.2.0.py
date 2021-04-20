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
    
    




# import the pkl file
#pkl_file=pd.read_pickle('/Users/bruce/Documents/uOttawa/Project/audio_brainstem_response/Data_BruceSunMaster_Studies/study2/study2DataFrame.pkl')
df_EFR=pd.read_pickle('/Users/bruce/Google Drive/Project/2.Code/df_EFR.pkl')


# Time domain


# average the df_EFR
df_EFR_avg = pd.DataFrame()
# average test1 and test2
for i in range(704):
    # combine next two rows later
    df_EFR_avg_t = pd.DataFrame(df_EFR.iloc[2*i: 2*i+2, 0:1024].mean(axis=0).values.reshape(1,1024)) # average those two rows
    df_EFR_label = pd.DataFrame(df_EFR.iloc[2*i, 1024:1031].values.reshape(1,7))
    df_EFR_avg = df_EFR_avg.append(pd.concat([df_EFR_avg_t, df_EFR_label], axis=1, ignore_index=True))

# set the title of columns
df_EFR_avg.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_EFR_avg = df_EFR_avg.sort_values(by=["Condition", "Subject"])
df_EFR_avg = df_EFR_avg.reset_index(drop=True)



# average all the subjects , test and retest, different sound levels
# filter by 'a vowel and 85Db'
df_EFR_avg_sorted = df_EFR_avg.sort_values(by=["Vowel", "Subject", "Condition", "Sound Level"])
df_EFR_avg_sorted = df_EFR_avg_sorted.reset_index(drop=True)
# average subjects, conditions, sound levels
df_EFR_avg_aenu = pd.DataFrame()
for i in range(4):
    # combine next two rows later
    df_EFR_avg_t = pd.DataFrame(df_EFR_avg_sorted.iloc[176*i: 176*i+176, 0:1024].mean(axis=0).values.reshape(1,1024)) # average those two rows
    df_EFR_avg_label = pd.DataFrame(df_EFR_avg_sorted.iloc[176*i, 1024:1031].values.reshape(1,7))
    temp = pd.concat([df_EFR_avg_t, df_EFR_avg_label], axis=1, ignore_index=True)
    df_EFR_avg_aenu = df_EFR_avg_aenu.append(temp, ignore_index=True)

# set the title of columns
df_EFR_avg_aenu.columns = np.append(np.arange(1024), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])
df_EFR_avg_aenu = df_EFR_avg_aenu.sort_values(by=["Condition", "Subject"])
#df_EFR_avg_aenu = df_EFR_avg_aenu.reset_index(drop=True)




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

n2 = 9606
k2 = np.arange(n2)
T2 = n2/sampling_rate
frq2 = k2/T2
freq2 = frq2[range(int(n2/2))]



# zero padding

# for df_EFR
df_EFR_data = df_EFR.iloc[:, :1024]
df_EFR_label = df_EFR.iloc[:, 1024:]
df_EFR_mid = pd.DataFrame(np.zeros((1408, 8582)))
df_EFR_withzero = pd.concat([df_EFR_data, df_EFR_mid, df_EFR_label], axis=1)
# rename columns
df_EFR_withzero.columns = np.append(np.arange(9606), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])

# for df_EFR_avg_aenu
df_EFR_avg_aenu_data = df_EFR_avg_aenu.iloc[:, :1024]
df_EFR_avg_aenu_label = df_EFR_avg_aenu.iloc[:, 1024:]
df_EFR_avg_aenu_mid = pd.DataFrame(np.zeros((4, 8582)))
df_EFR_avg_aenu_withzero = pd.concat([df_EFR_avg_aenu_data, df_EFR_avg_aenu_mid, df_EFR_avg_aenu_label], axis=1)
# rename columns
df_EFR_avg_aenu_withzero.columns = np.append(np.arange(9606), ["Subject", "Sex", "Condition", "Vowel", "Sound Level", "Num", "EFR/FFR"])







'''
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
markers2 = [100, 200, 300, 400, 500, 600, 700]
# which corresponds to 100 200....700Hz in frequency domain
plt.plot(temp_amplitude_spectrum_2, '-D', markevery=markers2)
plt.xlim(0, 1000)
# plt.xscale('linear')
plt.title('with zero padding')
plt.show()
'''

# Calculate the Amplitude Spectrum

# create a new dataframe with zero-padding amplitude spectrum

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


# for df_EFR_avg_aenu_withzero
df_aenu_as= pd.DataFrame()
df_aenu_as7= pd.DataFrame()
for i in range(4):
    #temp_aenu_EFR = df_EFR_avg_aenu_withzero.iloc[i, 0:9606]
    temp_aenu_as = np.abs((fftpack.fft(df_EFR_avg_aenu_withzero.iloc[i, 0:9606])/n2)[range(int(n2/2))])
    df_aenu_as = df_aenu_as.append(pd.DataFrame(temp_aenu_as.reshape(1,4803)), ignore_index = True)
    df_aenu_as7 = df_aenu_as7.append(pd.DataFrame(np.array([temp_aenu_as[100], temp_aenu_as[200], temp_aenu_as[300], temp_aenu_as[400], \
                                                            temp_aenu_as[500], temp_aenu_as[600], temp_aenu_as[700]]).reshape(1,7)), ignore_index = True)

df_aenu_as = pd.concat([df_aenu_as, df_EFR_avg_aenu_label], axis=1) # add labels on it
df_aenu_as7 = pd.concat([df_aenu_as7, df_EFR_avg_aenu_label], axis=1) # add labels on it


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




# plot the figure

# Time Domain and Frequency Domain

# grand mean aenu vowels 

# test
plt.figure()
plt.plot(np.asarray(df_EFR_avg_aenu.iloc[0,:1024]))
plt.plot(np.asarray(df_EFR_avg_aenu.iloc[1,:1024]))
plt.plot(np.asarray(df_EFR_avg_aenu.iloc[2,:1024]))
plt.plot(np.asarray(df_EFR_avg_aenu.iloc[3,:1024]))
plt.title('grand mean 4 vowel signal in time domain')
plt.show()

plt.figure()
#markers = [100, 200, 300, 400, 500, 600, 700] # which corresponds to 100 200....700Hz in frequency domain
plt.plot(df_aenu_as.iloc[0, :4803], '-')
plt.plot(df_aenu_as.iloc[1, :4803], '-')
plt.plot(df_aenu_as.iloc[2, :4803], '-')
plt.plot(df_aenu_as.iloc[3, :4803], '-')
plt.title('grand mean 4 vowel signal in frequency domain')
plt.xlim(0,1000)
plt.show()



# a vowel
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.asarray(df_EFR_avg_aenu.iloc[0,:1024]))
plt.title('grand mean a vowel signal in time and frequency domain')
plt.subplot(2,1,2)
markers = [100, 200, 300, 400, 500, 600, 700] # which corresponds to 100 200....700Hz in frequency domain
plt.plot(df_aenu_as.iloc[0, :4803], '-D', markevery=markers)
#plt.title('grand mean a vowel signal in frequency domain')
#xlabels=['100','200','300','400','500','600','700','800', '900', '1000']
#plt.set_xticks(np.arange(len(xlabels)))   
plt.xlim(0, 1000)
plt.show()

# e vowel
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.asarray(df_EFR_avg_aenu.iloc[1,:1024]))
plt.title('grand mean e vowel signal in time and frequency domain')
plt.subplot(2,1,2)
markers = [100, 200, 300, 400, 500, 600, 700] # which corresponds to 100 200....700Hz in frequency domain
plt.plot(df_aenu_as.iloc[1, :4803], '-D', markevery=markers)
#plt.title('grand mean e vowel signal in frequency domain')
#xlabels=['100','200','300','400','500','600','700','800', '900', '1000']
#plt.set_xticks(np.arange(len(xlabels)))   
plt.xlim(0, 1000)
plt.show()

# n vowel
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.asarray(df_EFR_avg_aenu.iloc[2,:1024]))
plt.title('grand mean n vowel signal in time and frequency domain')
plt.subplot(2,1,2)
markers = [100, 200, 300, 400, 500, 600, 700] # which corresponds to 100 200....700Hz in frequency domain
plt.plot(df_aenu_as.iloc[2, :4803], '-D', markevery=markers)
#plt.title('grand mean n vowel signal in frequency domain')
#xlabels=['100','200','300','400','500','600','700','800', '900', '1000']
#plt.set_xticks(np.arange(len(xlabels)))   
plt.xlim(0, 1000)
plt.show()

# u vowel
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.asarray(df_EFR_avg_aenu.iloc[3,:1024]))
plt.title('grand mean u vowel signal in time and frequency domain')
plt.subplot(2,1,2)
markers = [100, 200, 300, 400, 500, 600, 700] # which corresponds to 100 200....700Hz in frequency domain
plt.plot(df_aenu_as.iloc[3, :4803], '-D', markevery=markers)
#plt.title('grand mean u vowel signal in frequency domain')
#xlabels=['100','200','300','400','500','600','700','800', '900', '1000']
#plt.set_xticks(np.arange(len(xlabels)))   
plt.xlim(0, 1000)
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

