#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("english.wav")
mfcc_feat = mfcc(sig,rate)
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(sig,rate)
print(len(fbank_feat))
print(len(fbank_feat[0]))
print(fbank_feat[0][0])
# print(fbank_feat[:,:])
# print(fbank_feat[1:3,:])