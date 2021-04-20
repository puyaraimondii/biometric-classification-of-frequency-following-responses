#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:49:58 2019

@author: bruce
"""

### load module
from sklearn import datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from warnings import warn
import tensorflow as tf
import numpy as np
import os
import sys

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)

def get_file_name(path):
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            # print(root) #current path
            # print(dirs) #sub directories in current path
            # print(files) #all files in this path, directories not included
            return files
    return None

def file_filter(file_path, suffix):
    lst = get_file_name(file_path)
    if lst:
        return [item for item in lst if item.find(suffix) != -1]
    print('Path Not Exists!')
    
    
    
flags = tf.app.flags
flags.DEFINE_string('data_path_retest', '/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_avg_800_4_rename/retest/', 'data files path')
flags.DEFINE_string('data_path_test', '/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_avg_800_4_rename/test/', 'data files path')
# for mac os
# flags.DEFINE_string('record', '/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/EFR_85_spectrum_v008.tfrecord', 'record output path')
FLAGS = flags.FLAGS


def encode(encode_path):
    names = file_filter(encode_path, 'txt')
    num = 0
    file_none_lst = []
    data_output =[]
    label_output = []
    for name in names:
        if not os.path.getsize(encode_path+name):
            file_none_lst.append(name)
            continue
        data = np.loadtxt(encode_path+name)
        array_type = data.dtype.name
        array_shape = data.shape
        data = data.flatten()

        # row, col = data.shape
        #data = data / np.max(data)
        # data = data.reshape((row, col, 1))

        # label -> EFR_85_t_??_1.txt

        # set up label
        label_index = int(name[9:11])

        # print('label_index: ', label_index)

        # print ('data.shape', data.shape)
        # print ('label shape: ', label.shape)
        # data = np.reshape(data, (1,198))
        # print ('data.shape', data.shape)
        data_output.append(data)
        label_output.append(label_index)
        
        
        num += 1

    data_output = np.matrix(data_output)

    print('Total txt number: %s' % num)
    for file in file_none_lst:
        warn(file + ' is None.')
    return data_output, label_output



### load datasets
digits = datasets.load_digits()

#for ABR
data_r, label_r = encode(FLAGS.data_path_retest)
data_t, label_t = encode(FLAGS.data_path_test)

label_r = np.asarray(label_r)
label_t = np.asarray(label_t)
# digits_abr = {'data': data, 'target':label}

'''




'''
### data analysis
# print(digits.data.shape) 
# print(digits.target.shape)

### data split
'''
x_train,x_test,y_train,y_test = train_test_split(data_r,
                                                 label_r,
                                                 test_size = 0.99,
                                                 random_state = 3)
'''
# random_state =3 accuracy 57.14%


x_train = data_r
y_train = label_r
x_test = data_t
y_test = label_t


### fit model for train data
model = XGBClassifier()
model.fit(x_train,y_train)

### make prediction for test data
y_pred = model.predict(x_test)

### model evaluate
accuracy = accuracy_score(y_test,y_pred)
print("accuarcy: %.2f%%" % (accuracy*100.0))
print (classification_report(y_test, y_pred))