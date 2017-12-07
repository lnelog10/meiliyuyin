#!/bin/sh

#input filename
VIDEO_NAME="0001"
VIDEO_TRANS_IMAGE_PY="video_facial_landmarks_align_and_crop.py"
PREDICTOR_DAT="shape_predictor_68_face_landmarks.dat"

AUDIO_TRANS_PY="dealWidthAudio.py"

#mkdir 
workon cv
mkdir $VIDEO_NAME
ffmpeg -i "$VIDEO_NAME.mp4" -f mp3 -vn "$VIDEO_NAME/$VIDEO_NAME.mp3"
cp $VIDEO_NAME.mp4 $VIDEO_NAME

#cd to the dir
cd $VIDEO_NAME

#deal with 
python3 ../$AUDIO_TRANS_PY -i $VIDEO_NAME
python3 ../$VIDEO_TRANS_IMAGE_PY -i $VIDEO_NAME -p ../$PREDICTOR_DAT

rm *mp3
rm *mp4

#back to upper dir
cd ..






