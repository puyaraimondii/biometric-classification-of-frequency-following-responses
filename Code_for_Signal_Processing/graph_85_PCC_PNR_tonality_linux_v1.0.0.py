#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:59:54 2019

@author: bruce
"""

import numpy as np
import matplotlib.pyplot as plt


def threshold_bar(subject_num, subject_ov, subject_un, subject_tf_ov, subject_tf_un, label, title_1, title_2):
    # shown below is a example 
    """
    subject_num = 7
    subject_ov = (16, 15, 15, 14, 12, 8, 4)
    subject_un = (6, 3, 2, 2, 0, 0, 0)
    subject_tf_ov = (17, 16, 16, 15, 12, 8, 4)
    subject_tf_un = (5, 2, 1, 1, 0, 0, 0)
    label = ['>0', '>0.1', '>0.2', '>0.3', '>0.4', '>0.5', '>0.58']
    title_1 = 'subjects with all PCC score over threshold on time domain(Time Matrix)'
    title_2 = 'subjects with all PCC score over threshold on time domain(Time+Freq. Matrix)'
    """
    ind = np.arange(subject_num)
    width = 0.5
    
    plt.figure()
    
    ax1 = plt.subplot(121)
    p11 = ax1.bar(ind, subject_un, width, edgecolor='white')
    p12 = ax1.bar(ind, subject_ov, width, bottom=subject_un, edgecolor='white')
    
    ax2 = plt.subplot(122)
    p21 = ax2.bar(ind, subject_tf_un, width, edgecolor='white')
    p22 = ax2.bar(ind, subject_tf_ov, width, bottom=subject_tf_un, edgecolor='white')
    
    for r1, r2 in zip(p11, p12):
        h1 = r1.get_height()
        h2 = r2.get_height()
        ax1.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%d" % h1,
                 ha="center", va="bottom", color="white", fontsize=16, fontweight="bold")
        ax1.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%d" % h2,
                 ha="center", va="bottom", color="white", fontsize=16, fontweight="bold")
    
    for r1, r2 in zip(p21, p22):
        h1 = r1.get_height()
        h2 = r2.get_height()
        ax2.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%d" % h1,
                 ha="center", va="bottom", color="white", fontsize=16, fontweight="bold")
        ax2.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%d" % h2,
                 ha="center", va="bottom", color="white", fontsize=16, fontweight="bold")
    
    ax1.set_ylabel('Number of Subjects')
    ax1.set_xlabel('threshold value')
    ax1.title.set_text(title_1)
    ax1.set_xticks(np.arange(len(label)))
    ax1.set_xticklabels(label)
    ax1.set_yticks(np.arange(0, 23, 22))
    ax1.legend((p11[0], p12[0]), ('False-Predicted in Matrix', 'True-Predicted in Matrix'))
    
    ax2.set_ylabel('Number of Subjects')
    ax2.set_xlabel('threshold value')
    ax2.title.set_text(title_2)
    ax2.set_xticks(np.arange(len(label)))
    ax2.set_xticklabels(label)
    ax2.set_yticks(np.arange(0, 23, 22))
    ax2.legend((p21[0], p22[0]), ('False-Predicted in Matrix', 'True-Predicted in Matrix'))
    # plt.tight_layout()
    plt.show()

# logic 1
# if any 1 of 8 is higher than threshold -> accept
# otherwise (none of 8 higher than threshold) -> reject

# PCC_time


threshold_bar(subject_num=7,
              subject_ov=(16, 15, 15, 14, 12, 8, 4), 
              subject_un=(6, 3, 2, 2, 0, 0, 0), 
              subject_tf_ov=(17, 16, 16, 15, 12, 8, 4), 
              subject_tf_un=(5, 2, 1, 1, 0, 0, 0), 
              label=['>0', '>0.1', '>0.2', '>0.3', '>0.4', '>0.5', '>0.58'], 
              title_1='subjects with all PCC score over threshold on time domain(Time Matrix)', 
              title_2='subjects with all PCC score over threshold on time domain(Time+Freq. Matrix)'
              )


# PCC_ frequency

threshold_bar(subject_num=5, 
              subject_ov=(16, 14, 10, 9, 3), 
              subject_un=(6, 6, 4, 1, 0), 
              subject_tf_ov=(17, 17, 13, 10, 3), 
              subject_tf_un=(5, 3, 1, 0, 0), 
              label=['>0.5', '>0.6', '>0.7', '>0.8', '>0.84'],
              title_1='subjects with all PCC score over threshold on frequency domain(Freq Matrix)', 
              title_2='subjects with all PCC score over threshold on frequency domain(Time+Freq. Matrix)'
              )


# Peak Noise Ratio

threshold_bar(subject_num=10,
              subject_ov=(14, 12, 8, 7, 3, 2, 1, 1, 0, 0),
              subject_un=(6, 4, 1, 1, 1, 1, 0, 0, 0, 0),
              subject_tf_ov=(15, 15, 9, 8, 4, 3, 1, 1, 0, 0),
              subject_tf_un=(5, 1, 0, 0, 0, 0, 0, 0, 0, 0),
              label=['>1', '>2', '>3', '>4', '>5', '>6', '>7', '>8', '>9', '>9.7253'],
              title_1='subjects with all Peak Noise Ratio over threshold on frequency domain(Freq Matrix)',
              title_2='subjects with all Peak Noise Ratio over threshold on frequency domain(Time+Freq. Matrix)'
              )


# Tonality Coefficient
# 0 - 800 Hz
threshold_bar(subject_num=7,
              subject_ov=(13, 11, 10, 5, 2, 2, 2),
              subject_un=(7,  6,  6,  4, 4, 1, 1),
              subject_tf_ov=(15, 13, 12, 7, 4, 3, 3),
              subject_tf_un=(5,  4,  4,  2, 2, 0, 0),
              label=['<0.9', '<0.8', '<0.7', '<0.6', '<0.5', '<0.4', '<0.388'],
              title_1='subjects with all Tonality Coefficient over threshold on frequency domain(800Hz, Freq Matrix)',
              title_2='subjects with all Tonality Coefficient over threshold on frequency domain(800Hz, Time+Freq. Matrix)'
              )


# 0 - 1300 Hz
threshold_bar(subject_num=5,
              subject_ov=(10, 8, 3, 2, 1),
              subject_un=(6,  6, 4, 2, 1),
              subject_tf_ov=(12, 11, 5, 3, 2),
              subject_tf_un=(4,  3,  2, 1, 0),
              label=['<0.8', '<0.7', '<0.6', '<0.5', '<0.424'],
              title_1='subjects with all Tonality Coefficient over threshold on frequency domain(1300Hz, Freq Matrix)',
              title_2='subjects with all Tonality Coefficient over threshold on frequency domain(1300Hz, Time+Freq. Matrix)'
              )


# logic 2
# if all of 8  higher than threshold -> accept
# otherwise (anyone of 8 lower than threshold) -> reject

# PCC_time

threshold_bar(subject_num=5,
              subject_ov=(15, 15, 13, 9, 2),
              subject_un=(2,  2,  1,  0, 0),
              subject_tf_ov=(16, 16, 13, 9, 2),
              subject_tf_un=(1,  1,  1,  0, 0),
              label=['>0.58', '>0.6', '>0.7', '>0.8', '>0.9'],
              title_1='subjects with anyone of PCC score over threshold on time domain(Time Matrix)',
              title_2='subjects with anyone of PCC score over threshold on time domain(Time+Freq. Matrix)',
              )


# PCC_ frequency

threshold_bar(subject_num=3,
              subject_ov=(14, 13, 7),
              subject_un=(5,  2,  1),
              subject_tf_ov=(17, 14, 8),
              subject_tf_un=(2,  1,  0),
              label=['>0.84', '>0.9', '>0.95'],
              title_1='subjects with anyone of PCC score over threshold on frequency domain(Freq Matrix)',
              title_2='subjects with anyone of PCC score over threshold on frequency domain(Time+Freq. Matrix)'
              )


# Peak Noise Ratio

threshold_bar(subject_num=11,
              subject_ov=(13, 11, 11, 10, 10, 8, 8, 8, 4, 3, 3),
              subject_un=(3,  1,  1,  1,  1,  1, 1, 1, 1, 1, 0),
              subject_tf_ov=(15, 12, 12, 11, 11, 9, 9, 9, 5, 4, 3),
              subject_tf_un=(1,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0),
              label=['>7', '>8', '>9', '>9.7253', '>10', '>11', '>12', '>13', '>14', '>15', '16'],
              title_1='subjects with anyone of Peak Noise Ratio over threshold on frequency domain(Freq Matrix)',
              title_2='subjects with anyone of Peak Noise Ratio over threshold on frequency domain(Time+Freq. Matrix)',
              )


# Tonality Coefficient
# 0 - 800 Hz
threshold_bar(subject_num=4,
              subject_ov=(15, 14, 8, 5),
              subject_un=(5,  5,  4, 0),
              subject_tf_ov=(16, 15, 9, 5),
              subject_tf_un=(4,  4,  3, 0),
              label=['<0.388', '<0.3', '<0.2', '<0.1'],
              title_1='subjects with anyone of Tonality Coefficient over threshold on frequency domain(800Hz, Freq Matrix)',
              title_2='subjects with anyone of Tonality Coefficient over threshold on frequency domain(800Hz, Time+Freq. Matrix)'
              )

# 0 - 1300 Hz
threshold_bar(subject_num=5,
              subject_ov=(14, 14, 14, 8, 5),
              subject_un=(5,  5,  5,  5, 0),
              subject_tf_ov=(15, 15, 15, 10, 5),
              subject_tf_un=(4,  4,  4,  3,  0),
              label=['<0.424', '<0.4', '<0.3', '<0.2', '<0.1'],
              title_1='subjects with anyone of Tonality Coefficient over threshold on frequency domain(1300Hz, Freq Matrix)',
              title_2='subjects with anyone of Tonality Coefficient over threshold on frequency domain(1300Hz, Time+Freq. Matrix)'
              )

# logic 3
# average ove threshold
# PCC_time

threshold_bar(subject_num=2, 
              subject_ov=(13, 13), 
              subject_un=(1, 0), 
              subject_tf_ov=(13, 13), 
              subject_tf_un=(1, 0), 
              label=['>0.58', '>0.6'],
              title_1='subjects with average PCC score over threshold on time domain(Time Matrix)', 
              title_2='subjects with average PCC score over threshold on time domain(Time+Freq. Matrix)'
              )


# PCC_ frequency

threshold_bar(subject_num=3, 
              subject_ov=(11, 8, 5), 
              subject_un=(2,  1, 0), 
              subject_tf_ov=(12, 9, 5), 
              subject_tf_un=(1,  0, 0), 
              label=['>0.84', '>0.875', '>0.9'],
              title_1='subjects with average PCC score over threshold on frequency domain(Freq Matrix)', 
              title_2='subjects with average PCC score over threshold on frequency domain(Time+Freq. Matrix)'
              )


# Peak Noise Ratio

threshold_bar(subject_num=7,
              subject_ov=(13, 12, 10, 8, 5, 3, 3),
              subject_un=(4,  1,  1,  1, 1, 1, 1),
              subject_tf_ov=(16, 13, 11, 9, 6, 4, 4),
              subject_tf_un=(1, 0, 0, 0, 0, 0, 0),
              label=['>4', '>5', '>6', '>7', '>8', '>9', '>9.7253'],
              title_1='subjects with average Peak Noise Ratio over threshold on frequency domain(Freq Matrix)',
              title_2='subjects with average Peak Noise Ratio over threshold on frequency domain(Time+Freq. Matrix)'
              )


# Tonality Coefficient
# 0 - 800 Hz
threshold_bar(subject_num=3,
              subject_ov=(7, 2, 1),
              subject_un=(5, 2, 0),
              subject_tf_ov=(9, 3, 1),
              subject_tf_un=(3, 1, 0),
              label=['<0.388', '<0.3', '<0.2'],
              title_1='subjects with average Tonality Coefficient over threshold on frequency domain(800Hz, Freq Matrix)',
              title_2='subjects with average Tonality Coefficient over threshold on frequency domain(800Hz, Time+Freq. Matrix)'
              )

# 0 - 1300 Hz
threshold_bar(subject_num=4,
              subject_ov=(7, 7, 2, 1),
              subject_un=(5, 5, 1, 0),
              subject_tf_ov=(9, 9, 3, 1),
              subject_tf_un=(3, 3, 0, 0),
              label=['<0.424', '<0.4', '<0.3', '<0.2'],
              title_1='subjects with average Tonality Coefficient over threshold on frequency domain(1300Hz, Freq Matrix)',
              title_2='subjects with average Tonality Coefficient over threshold on frequency domain(1300Hz, Time+Freq. Matrix)',
              )
