#!/bin/sh

for file in * 
do 
	if test -f $file
	then 
		if [[ $file =~ \.mp4$ ]] 
		then 
			echo "deal $file"
			basename=${file%.*}
			echo "basename $basename"
			source dealWithOneMp4.sh $basename
		fi
	fi
done
