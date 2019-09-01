#!/usr/bin/python3
# -*- coding: utf-8 -*-


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import scipy.io.wavfile as wav
from python_speech_features import *
import numpy as np
import sklearn.preprocessing
from yuyin import test_main
from stop_record import Recorder
import time
import speech_lib


path_film = os.path.abspath('.')
path = path_film + "/data/xunlian/"
test_path = path_film + "/data/test_data/"
isnot_test_path = path_film + "/data/isnot_test_path/"
gym_test_path = path_film + "/data/gym_test_path/"

def main():
    rec = Recorder()
    print ("Prepare ...")
    time.sleep(1)
    rec.record("0_0.wav", gym_test_path)
    test_main(gym_test_path)

if __name__ == '__main__':

    main()
    

