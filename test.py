#!/usr/bin/python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import scipy.io.wavfile as wav
from python_speech_features import mfcc,delta
import os
import numpy as np
import sklearn.preprocessing
from matplotlib import pyplot as plt
from matplotlib.pyplot import specgram 
from matplotlib.pyplot import cm
import speech_lib

plt.ion()

path_film = os.path.abspath('.')
path = path_film + "/data/xunlian/"
test_path = path_film + "/data/test_data/"
isnot_test_path = path_film + "/data/isnot_test_path/"



def read_wav_path(path):

    map_path, map_relative = [str(path) + str(x) for x in os.listdir(path) if os.path.isfile(str(path) + str(x))], [y for y in os.listdir(path)]
    return map_path, map_relative


########################################################
def plot_wav(fs, audio):
    frames = audio.shape
    time = np.arange(0, frames[0]) * (1.0/fs)
    plt.plot(time, audio)
 

def plot_mfcc(mfcc_data):
    fig, ax = plt.subplots()
    mfcc_data = processed_audio
    mfcc_data = np.swapaxes(mfcc_data, 0, 1)
    cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')

def list2txt(npdata):
    file = open('data.txt', 'w')
    for fp in npdata:
        file.write(str(fp)+' ')
    file.close()


if __name__ == '__main__':
    map_path, map_relative = read_wav_path(path)

    #print (map_path)
    for i in range(1):
        #fs, audio = wav.read(map_path[i]) #对这个文件夹下的所有wav文件绘图，就使用注释部分
        fs, audio = wav.read('/home/michael/Workspace/gym_speech_recog_eng/data/gym_test_path/0_0.wav') #读取wav文件
        audio = audio[...,0]
        print ("fs: "+str(fs))
        #file_n = (map_path[i]).split('/')
        #file_n = file_n[-1]
        file_n = '0_0.wav'
        print (audio.shape)
        print (audio)
        list2txt(audio)

        plt.figure()
        plot_wav(fs,audio) #画声波图
        plt.title('Wave_'+str(file_n))


        processed_audio = mfcc(audio, samplerate=fs, nfft=2000)
        #plot_mfcc(processed_audio)
        #plt.figure()
        plt.matshow(processed_audio.T) #画MFCC矩阵图
        plt.title('MFCC_'+str(file_n))


    plt.ioff()
    plt.show()
    
