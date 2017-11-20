#!/bin/sh
cat urls.txt | while read line
do 
	cd audio/
	video_url="https://www.youtube.com/watch?v=$line"
	echo "downd videod's url:$video_url"
	#youtube-dl -f bestaudio "$video_url"
	youtube-dl -f bestvideo "$video_url"
	cd ../
done

cd audio
for file in *
do 
	if [ -f "$file" ]
	then 
		echo  "now decode $file"
		#ffmpeg -i $file -acodec libmp3lame -ab 128k ../for_mp3/$file.mp3
		#ffmpeg -i ../for_mp3/$file.mp3 -acodec pcm_u8 -ar 22050 ../for_wav/$file.wav 
		ffmpeg -i "$file" "../for_wav/$file.wav"
	fi
done



