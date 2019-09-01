#!/usr/bin/python3
# -*- coding: utf-8 -*-

import scipy.io.wavfile 
import matplotlib.pyplot as plt 
import urllib2 
import numpy as np
import scipy.signal
from python_speech_features import mfcc,delta
import heapq

KEY = 100

def filt(indata):
    b, a = scipy.signal.butter(8, 0.05, 'highpass')  
    filtered = scipy.signal.filtfilt(b, a, indata) 
    return filtered

def adjust_vol(data):
    re = heapq.nlargest(500, data)
    key = np.mean(re)
    print ("key: "+str(key))
    return data



def plot_wave(data):
    plt.figure()
    plt.plot(data)

def plot_mfcc(data):
    processed_audio = mfcc(data, samplerate=fs, nfft=2000)
    plt.matshow(processed_audio.T)
    plt.title('MFCC_0')

if __name__ == '__main__':
        
    # 使用 SciPy 读取音频文件
    fs, data = scipy.io.wavfile.read('/home/michael/Workspace/gym_speech_recog_eng/data/gym_test_path/0_0.wav') 
    print("Data type", data.dtype, "Shape", data.shape)
    # ('Data type', dtype('uint8'), 'Shape', (43584L,))
    data = data[...,0]

    # 绘制原始音频文件
    plt.figure()
    plt.title("Original") 
    plt.plot(data)

    processed_audio = mfcc(data, samplerate=fs, nfft=2000)
    plt.matshow(processed_audio.T)
    plt.title('MFCC_0')

    # 设计滤波器，iirdesign 设计无限脉冲响应滤波器
    # 参数依次是 0 ~ 1 的正则化频率、
    # 最大损失、最低衰减和滤波类型
    #b,a = scipy.signal.iirdesign(wp=0.2, ws=0.1, gstop=60, gpass=1, ftype='butter')

    b, a = scipy.signal.butter(8, 0.05, 'highpass')  
    filtered = scipy.signal.filtfilt(b, a, data)  

    # 传入刚才的返回值，使用 lfilter 函数来调用滤波器
    #filtered = scipy.signal.lfilter(b, a, data)
    #filtered = filtered*2

    # 绘制滤波后的音频
    plt.figure()
    plt.title("Filtered") 
    plt.plot(filtered)

    processed_audio = mfcc(filtered, samplerate=fs, nfft=2000)
    plt.matshow(processed_audio.T)
    plt.title('MFCC_1')

    # 保存滤波后的音频
    scipy.io.wavfile.write('/home/michael/Workspace/gym_speech_recog_eng/data/gym_test_path/0_1.wav', fs, filtered. astype(data.dtype))
    plt.show()


