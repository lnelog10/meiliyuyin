from __future__ import print_function
import numpy as np
import scipy
import matplotlib.pyplot as plt
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment

# song = AudioSegment.from_wav("audio/Zach Galifianakis & Isla Fisher Exclusive Interview - Keeping Up with the Joneses-rRU7RygkMDA.webm.wav")
song = AudioSegment.from_wav("audio/Keeping Up with the Joneses Interview - Isla Fisher (2016) - Comedy-O9rCpfpXtyY.m4a.wav")
print(song.__len__())
# print(song.duration_seconds)
sum = int(song.__len__()/350)
print(sum)
# if (350*sum() < song.__len__()):
# sum = sum+1 最后不能整除的丢弃
dirname = "audio/Keeping Up with the Joneses Interview/"
for i in range(sum):
    next = (i + 1) * 350
    first_10_seconds = song[i * 350:next]
    first_10_seconds.export(dirname+str(i)+".wav", format="wav")
    y1, sr1 = librosa.load(dirname+str(i)+".wav", sr=16000)
    print(len(y1))
    mfccs = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13, hop_length=164, n_fft=2048)#13*35
    print(len(mfccs).__str__()+"*"+len(mfccs[0]).__str__())
    print(type(mfccs))
    np.savetxt(dirname+str(i)+".txt",mfccs)
    print(str(i)+"----------------------------")


