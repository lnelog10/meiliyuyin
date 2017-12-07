#!/bin/sh

index=0
for file in * 
do 
	if test -f $file
	then 
		if [[ $file =~ \.mp4$ ]] 
		then 
			let "index=$index+1"
			basename=`seq -f "%04g" $index $index`
			echo "move $file to $basename.mp4"
			mv $file "$basename.mp4"
		fi
	fi
done
