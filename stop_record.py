#!/usr/bin/python3
# -*- coding: utf-8 -*-
 
import pyaudio
import time
import threading
import wave
import os

path_film = os.path.abspath('.')
path = path_film + "/data/xunlian/"
test_path = path_film + "/data/test_data/"
isnot_test_path = path_film + "/data/isnot_test_path/"
 
class Recorder():
    def __init__(self, chunk=1024, channels=2, rate=8000):
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
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self._frames))
        wf.close()
        print("Saved")

    def record(self, filename, the_path = isnot_test_path):
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
        

def rec():
    for i in range(0,40):
        a = int(input('请输入相应数字开始:'))
        if a == 1:           
            rec = Recorder()
            begin = time.time()
            print("Start recording")
            rec.start()
            b = int(input('请输入相应数字停止:'))
            if b == 2:
                print("Stop recording")
                rec.stop()
                fina = time.time()
                t = fina - begin
                print('录音时间为%ds'%t)
                rec.save("1_%d.wav"%i)

 
if __name__ == "__main__":
    
    for i in range(0,40):
        rec = Recorder()
        print ("Prepare %d..."%i)
        time.sleep(1)
        rec.record("4_%d.wav"%i)


