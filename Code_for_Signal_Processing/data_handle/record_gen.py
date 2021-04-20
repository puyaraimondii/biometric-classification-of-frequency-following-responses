# encoding utf-8


# Created:    on June 7, 2018 20:10
# @Author:    xxoospring

r"""tfrecords generate
########################
All wav file names should be in this format:
# wav files name:
#     0_*.wav:
#         0 represent class label
########################
Example usage:
    python record_gen.py --data_path=YOU_DATA_PATH --record=PATH/*.tfrecords
"""
from warnings import warn
import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('../pub/')
from file import *

flags = tf.app.flags
flags.DEFINE_string('data_path', ' ', 'data files path')
flags.DEFINE_string('record', ' ', 'record output path')
FLAGS = flags.FLAGS


def encode():
    names = file_filter(FLAGS.data_path, 'txt')
    num = 0
    file_none_lst = []
    with tf.python_io.TFRecordWriter(FLAGS.record) as tf_writer:
        for name in names:
            if not os.path.getsize(FLAGS.data_path+name):
                file_none_lst.append(name)
                continue
            data = np.loadtxt(FLAGS.data_path+name)
            data = data.flatten()
            # row, col = data.shape
            data = data / np.max(data)
            # data = data.reshape((row, col, 1))

            # label -> EFR_85_t_??_1.txt
            label = int(name[9:11])
            # print(data.shape)
            example = tf.train.Example(features=tf.train.Features(feature={
                'spec': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.tobytes()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            serialized = example.SerializeToString()
            tf_writer.write(serialized)
            num += 1
            print(num)
    # TODO(xxoospring) python interpreter collapse when this called
    # tf_writer.close()
    print('Total txt number: %s' % num)
    for file in file_none_lst:
        warn(file + ' is None.')


def main(_):
    if not os.path.exists(FLAGS.data_path):
        raise ValueError('data path not found!')
    encode()


if __name__ == '__main__':
    tf.app.run()