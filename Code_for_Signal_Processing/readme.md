{\rtf1\ansi\ansicpg1252\cocoartf1671\cocoasubrtf100
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 Readme\
\
\
For version 1.4.0\
\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\
1. Add new function: shrink_value_0.3_1\
This function is used to shrink the value of correlation coefficient from [-1, 1] to [0.3,1] in order to successfully use colormap -> scale to represent different value intervals.\
\
For example, \
previously, 10 scale for [-1, 1]\
Now, excited, 10 scale for [0.3, 1]\
\
\
V1.5.0\
\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\'97\
Solve the problem for def shrink_value_0.3_1 & shrink_value_0.5_1\
By using output = input.copy() instead of output = input\
Coz pandas.dataframe uses the same index if directly use output = input\
\
}