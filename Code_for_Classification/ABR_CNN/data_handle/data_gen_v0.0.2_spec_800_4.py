# encoding utf-8


# Created:    on March 11, 2019
# @Author:    Bruce

"""tfrecords generate
########################
All txt file names should be in this format:
# txt files name:
#     EFR_85_r_01_1.wav:
#         85 represent soudn level
#         r represent retest(r)/ test(t)
#         01 represent class label
#         1 represent number 1/2
########################
Example usage:
    python record_gen.py --data_path=YOUR_DATA_PATH --record=PATH/*.tfrecord
    python record_gen_v0.0.3.py --
"""
from warnings import warn
import tensorflow as tf
import numpy as np
import os
import sys

from pub.file import file_filter

sys.path.append('../pub/')
from file import *


flags = tf.app.flags
flags.DEFINE_string('data_path', '/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_800_4_rename/', 'data files path')
# for mac os
flags.DEFINE_string('record', '/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/EFR_85_spectrum_v009.tfrecord', 'record output path')
FLAGS = flags.FLAGS

save_path = '/home/bruce/Dropbox/Project/5.Code_Example/ABR_CNN/data/'

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
flags.DEFINE_string('data_path_retest', '/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_800_4_rename/retest/', 'data files path')
flags.DEFINE_string('data_path_test', '/home/bruce/Dropbox/Project/6.Result/data_spectrogram/EFR/85_800_4_rename/test/', 'data files path')
FLAGS = flags.FLAGS


def encode(encode_path):
    names = file_filter(encode_path, 'txt')
    num = 0
    file_none_lst = []
    data_list = []
    label_list = []
    for name in names:
        if not os.path.getsize(encode_path+name):
            file_none_lst.append(name)
            continue
        data = np.loadtxt(encode_path+name)
        data = data/np.max(data)
        array_type = data.dtype.name
        array_shape = data.shape
        data_list.append(data)
        # row, col = data.shape
        #data = data / np.max(data)
        # data = data.reshape((row, col, 1))

        # label -> EFR_85_t_??_1.txt

        # set up label

        label_index = int(name[9:11])
        label_index = label_index - 1
        label_list.append(label_index)
        num += 1
    # print('Total txt number: %s' % num)
    data_array = np.stack(data_list, axis=0)
    label_array = np.stack(label_list, axis=0)
    # print (data_array.shape)
    # print(label_array.shape)
    # print(label_array)
    for file in file_none_lst:
        warn(file + ' is None.')
    return data_array, label_array

data_r, label_r = encode(FLAGS.data_path_retest)
data_t, label_t = encode(FLAGS.data_path_test)



np.save('../data/spec_800_4/data_r', data_r)
np.save('../data/spec_800_4/label_r', label_r)
np.save('../data/spec_800_4/data_t', data_t)
np.save('../data/spec_800_4/label_t', label_t)
