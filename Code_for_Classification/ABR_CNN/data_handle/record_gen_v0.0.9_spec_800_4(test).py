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


def encode():
    names = file_filter(FLAGS.data_path, 'txt')
    num = 0
    file_none_lst = []
    data_list = []
    label_list = []
    for name in names:
        if not os.path.getsize(FLAGS.data_path+name):
            file_none_lst.append(name)
            continue
        data = np.loadtxt(FLAGS.data_path+name)
        array_type = data.dtype.name
        array_shape = data.shape
        data_list.append(data)
        data = data.flatten()

        data_bytes = data.tobytes()
        # row, col = data.shape
        #data = data / np.max(data)
        # data = data.reshape((row, col, 1))

        # label -> EFR_85_t_??_1.txt

        # set up label
        label = np.zeros(22)
        label_index = int(name[9:11])
        label_index = label_index - 1
        label_list.append(label_index)

        print('label_index: ', label_index)
        label[label_index] = 1
        print ('data.shape', data.shape)
        print ('label shape: ', label.shape)
        # print('label shape: ', label.shape)

        # print(data.shape)
        example = tf.train.Example(features=tf.train.Features(feature={
            'spec': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.tobytes()])),
            'label': tf.train.Feature(float_list=tf.train.FloatList(value=label))
        }))
        serialized = example.SerializeToString()
        tf_writer.write(serialized)
        num += 1
        '''
        print(num)
        print(data)
        print(type(data))
        print('array_type:', array_type)
        print('array_shape:', array_shape)
        print(np.frombuffer(data_bytes, dtype=array_type).reshape(array_shape))
        print((np.frombuffer(data_bytes, dtype=array_type).reshape(array_shape)).shape)
        '''


    # tf_writer.close()
    print('Total txt number: %s' % num)
    data_array = np.stack(data_list, axis=0)
    label_array = np.stack(label_list, axis=0)
    print (data_array.shape)
    print(label_array.shape)
    print(label_array)
    for file in file_none_lst:
        warn(file + ' is None.')


def main(_):
    if not os.path.exists(FLAGS.data_path):
        raise ValueError('data path not found!')
    encode()


if __name__ == '__main__':
    tf.app.run()