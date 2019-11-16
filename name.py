# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:22:38 2019

@author: shaomingtian
"""

import pandas as pd
from collections import defaultdict
import math

# 读取train.txt
train = pd.read_csv('train.txt',engine='python')




test = pd.read_csv('test.txt',engine='python')
submit = pd.read_csv('sample_submit.csv',engine='python')

#把数据分为男女两部分
names_female = train[train['gender'] == 0]
names_male = train[train['gender'] == 1]

#totals统计训练集男女的个数
totals = {'f': len(names_female),
          'm': len(names_male)}

#统计每个字在女性名字中出现的频率
frequency_list_f = defaultdict(int)
for name in names_female['name']:
    for char in name:
        frequency_list_f[char] += 1. / totals['f']

#统计每个字在男性名字中出现的频率
frequency_list_m = defaultdict(int)
for name in names_male['name']:
    for char in name:
        frequency_list_m[char] += 1. / totals['m']

#计算拉普拉斯平滑，这里的alpha取值为1
#运用拉普拉斯解决0事件问题
def Laplace(char, frequency_list, total, alpha=1.0):
    count = frequency_list[char] * total #char字出现的总次数
    distinct_chars = len(frequency_list) #总共不同字符的个数
    smooth = (count + alpha ) / (total + distinct_chars * alpha) #拉普拉斯公式
    return smooth

#对拉普拉斯的结果取对数，得到logP(X_i=0|Y=y)
def Log_Result(char, frequency_list, total):
    freq_smooth = Laplace(char, frequency_list, total)
    return math.log(freq_smooth) - math.log(1 - freq_smooth)
    #return freq_smooth

#得到y=0 y=1分别的可能性大小
def Compute_Log_Result(name, commons, totals, frequency_list_m, frequency_list_f):
    logprob_m = commons['m']
    logprob_f = commons['f']
    for char in name:
        logprob_m += Log_Result(char, frequency_list_m, totals['m'])
        logprob_f += Log_Result(char, frequency_list_f, totals['f'])
    return {'male': logprob_m, 'female': logprob_f}

def GetGender(LogProbs):
    return LogProbs['male'] > LogProbs['female']


common_f = math.log(1 - train['gender'].mean())
common_f += sum([math.log(1 - frequency_list_f[char]) for char in frequency_list_f])

common_m = math.log(train['gender'].mean())
common_m += sum([math.log(1 - frequency_list_m[char]) for char in frequency_list_m])

commons = {'f': common_f, 'm': common_m}


result = []
for name in test['name']:
    LogProbs = Compute_Log_Result(name, commons, totals, frequency_list_m, frequency_list_f)
    gender = GetGender(LogProbs)
    result.append(int(gender))

submit['gender'] = result

submit.to_csv('my_NB_prediction12.csv', index=False)

#莉亚  娜娜   丽丽    世银  明天
#建国  天  军  昌  泽欢  鹏

tt='建国'
div = Compute_Log_Result(tt, commons, totals, frequency_list_m, frequency_list_f)
print(div['male'] , div['female'])
gender = '男' if GetGender(div) else '女' 
print(gender)

