#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:52:08 2019

@author: bruce
"""

#coding=utf-8
from sklearn.datasets import  load_digits  # 从skleran.datasets里有导入手写体数字加载器
from sklearn.model_selection import  train_test_split #导入train_test_split用于数据分割
from sklearn.preprocessing import StandardScaler  # 导入标准化数据
from sklearn.svm import LinearSVC                       # 从sklearn.svm中导入基于现行假设的支持向量机分类器LinearSVC
from sklearn.metrics import classification_report
from sklearn import svm
import matplotlib.pyplot as plt

#1.数据获取
### load module
from sklearn import datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix

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
flags.DEFINE_string('data_path_retest', '/home/bruce/Dropbox/4.Project/6.Result/data_spectrogram/EFR/85_melspectrogram_22_33_rename/retest/', 'data files path')
flags.DEFINE_string('data_path_test', '/home/bruce/Dropbox/4.Project/6.Result/data_spectrogram/EFR/85_melspectrogram_22_33_rename/test/', 'data files path')
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
# digits = datasets.load_digits()

#for ABR
data_r, label_r = encode(FLAGS.data_path_retest)
data_t, label_t = encode(FLAGS.data_path_test)

label_r = np.asarray(label_r)
label_t = np.asarray(label_t)
# digits_abr = {'data': data, 'target':label}


### data split
'''
X_train,X_test,y_train,y_test = train_test_split(digits.data,
                                                 digits.target,
                                                 test_size=0.2,
                                                 random_state=33)

'''
X_train = data_r
y_train = label_r
X_test = data_t
y_test = label_t



# print (X_train)
# print (y_train)
# print (y_train.shape)
ss = StandardScaler()                    # 对测试集和训练集进行标准化
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

#3.使用SVM训练
# lsvc = LinearSVC()                      # 初始化现行假设的支持向量机分类器 LinearSVC
# lsvc.fit(X_train,y_train)                # 进行模型训练
# y_predict = lsvc.predict(X_test)

clf_lsvc = LinearSVC() 
clf_lsvc.fit(X_train,y_train)
score_lsvc = clf_lsvc.score(X_test,y_test)
y_predict = clf_lsvc.predict(X_test)
print("The score of lsvc is : %f"%score_lsvc)


# kernel = 'rbf'
clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(X_train,y_train)
score_rbf = clf_rbf.score(X_test,y_test)
pred_rbf = clf_rbf.predict(X_test)
print("The score of rbf is : %f"%score_rbf)
print ('classification_report of rbf')
print (classification_report(y_test, pred_rbf))  


# kernel = 'linear'
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(X_train,y_train)
score_linear = clf_linear.score(X_test,y_test)
pred_linear = clf_rbf.predict(X_test)
print("The score of linear is : %f"%score_linear)
print ('classification_report of linear')
print (classification_report(y_test, pred_linear))


# kernel = 'poly'
clf_poly = svm.SVC(kernel='poly')
clf_poly.fit(X_train,y_train)
score_poly = clf_poly.score(X_test,y_test)
pred_poly = clf_poly.predict(X_test)
print("The score of poly is : %f"%score_poly)
print ('classification_report of poly')
print (classification_report(y_test, pred_poly))



# plot confusion matrix

model=clf_rbf
disp = plot_confusion_matrix(model, X_test, y_test,
                                 #display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=None)


