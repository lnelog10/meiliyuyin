#!/bin/sh

for file in * 
do 
	if test -f $file
	then 
		if [[ $file =~ \.mp4$ ]] 
		then 
			basename=${file%.*}
			echo "deal with $basename.mp4"
			source dealWithOneMp4.sh $basename
		fi
	fi
done
