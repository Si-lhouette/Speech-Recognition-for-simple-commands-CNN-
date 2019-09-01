#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pyaudio
import time
import threading
import wave
import speech_recognition as sr
import os

path = os.path.abspath('.')
path = path + '/'

class Recorder():
    def __init__(self, chunk=1024, channels=1, rate=8000):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = True
        self._frames = []
        self.time = 2

    def start(self):
        threading._start_new_thread(self.__recording, ())

    def __recording(self):
        self._running = True
        self._frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        while(self._running):
            data = stream.read(self.CHUNK)
            self._frames.append(data)
 
        stream.stop_stream()
        stream.close()
        p.terminate()
 
    def stop(self):
        self._running = False
 
    def save(self, filename, the_path):
        
        p = pyaudio.PyAudio()
        if not filename.endswith(".wav"):
            filename = filename + ".wav"
        filename = the_path + filename
        print(filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self._frames))
        wf.close()
        print("Saved")

    def record(self, filename, the_path = path):
        begin = time.time()
        
        self.start()
        print("")
        print("Start recording")
        count = 0
        while count < self.time:
            count = time.time() - begin
            print '\r'+str(count),
        print ('time: '+ str(count))
        print("Stop recording")
        self.stop()
        self.save(filename, the_path)
        


if __name__ == '__main__':

    rec = Recorder()
    i=0
    print ("Prepare %d..."%i)
    time.sleep(1)
    rec.record("0_%d.wav"%i)

    # obtain audio from the microphone
    r = sr.Recognizer()

    audio = sr.AudioFile('0_0.wav')
    with audio as source:
        data = r.record(source)
        s = r.recognize_sphinx(data)
        print(s)

