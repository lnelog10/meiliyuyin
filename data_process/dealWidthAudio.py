from pydub import AudioSegment
import librosa
import numpy as np
import argparse

parse = argparse.ArgumentParser(description='')
parse.add_argument("-i","--input",required=True,help="input video")

def sample_voice_process(prefix, SampleVoice):
    song = AudioSegment.from_mp3(SampleVoice)
    sum = int(song.__len__()/350)
    # if (350*sum() < song.__len__()):
    # sum = sum+1 最后不能整除的丢弃
    for i in range(sum):
        next = (i + 1) * 350
        first_10_seconds = song[i * 350:next]
        index =(i + 1);
        mp3name = prefix + '{:04d}.mp3'.format(index)
        first_10_seconds.export( mp3name, format="mp3")
        y1, sr1 = librosa.load(mp3name, sr=16000)
        mfccs = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13, hop_length=192, n_fft=2048)#13*35
        print("==>"+str(i)+"<==",len(mfccs).__str__() + "*" + len(mfccs[0]).__str__())
        np.savetxt(prefix+'{:04d}.txt'.format(index),mfccs)

def main():
    args = vars(parse.parse_args())
    audioName = args["input"] + ".mp3"
    sample_voice_process(args["input"],audioName)

if __name__ == '__main__':
    main()