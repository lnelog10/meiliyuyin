import argparse
import os


def main():
    parse = argparse.ArgumentParser(description='')
    parse.add_argument("-i","--input",required=True,help="input video")

    args = vars(parse.parse_args())

    videoName = args["input"]
    prefix = getFilenameWithoutProfix(videoName)
    dir = prefix
    if not os.path.exists(dir):
        os.makedirs(dir)
    audioName = departAudioFromVideo(videoName, prefix)









def departAudioFromVideo(videoName,audioName):
    os.system("ffmpeg -i "+videoName+ " -f mp3 -vn "+audioName+".mp3")
    return audioName+".mp3"
        
    
    













    """
    取一个文件名的前缀
    """
def getFilenameWithoutProfix(filename):
    temp = filename.split('.')
    return temp[0]

if __name__ == '__main__':
    main()

