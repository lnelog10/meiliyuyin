from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math

ap = argparse.ArgumentParser()
#ap.add_argument("-i","--input", required=True, help="path to input video")
#ap.add_argument("-o","--output", required=True, help="path to output video")
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
args = vars(ap.parse_args())

cap = cv2.VideoCapture("../youtube_video/K.mp4")
#cap = cv2.VideoCapture(args["input"])
#cap = cv2.VideoCapture(0)

print("frame count:",cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("frame rate:",cap.get(cv2.CAP_PROP_FPS))


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter(args["output"],fourcc, 20.0, (640,480))
#out = cv2.VideoWriter("output.avi",fourcc, 20.0, (800,800))
out = None
(h,w) = (None,None)
(x,y,w,h) = (None,None,None,None)

time_count = -1
SLOT_TIME = 350

while(cap.isOpened()):
	cap.set(cv2.CAP_PROP_POS_MSEC,time_count * SLOT_TIME)
	time_count = time_count + 1
	ret, image = cap.read()
	if ret == True:
#		image = imutils.resize(image, width=500)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		rects = detector(gray,1)
		print("====================================================>")
		print("big image type:",image.dtype)
		print("gray image type:",gray.dtype)
		for(i, rect) in enumerate(rects):
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			(x,y,w,h) = face_utils.rect_to_bb(rect)
#			(x,y,w,h) = cv2.boundingRect(np.array([shape[i:j]]))
			#cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
#			cv2.putText(image, "genius #{}".format(i + 1),(x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),2)
			(x1,y1,w1,h1) = (x,y,w,h)

			(desireHeight,desireWidth) = image.shape[:2]
			desireHeight = desireWidth
			
#			for(x,y) in shape:
#				cv2.circle(image,(x,y),1,(0,0,255),-1)
			#(leftEyeOutY,leftEyeOutX) = shape[37]
			#(rightEyeOutY,rightEyeOutX) = shape[46]
			#(leftEyeInnerY,leftEyeInnerX) = shape[40]
			#(rightEyeInnerY,rightEyeInnerX) = shape[43]
			(leftEyeOutX,leftEyeOutY) = shape[36]
			(rightEyeOutX,rightEyeOutY) = shape[45]
			(leftEyeInnerX,leftEyeInnerY) = shape[39]
			(rightEyeInnerX,rightEyeInnerY) = shape[42]
			
			leftEyeY = (leftEyeOutY + leftEyeInnerY)/2
			leftEyeX = (leftEyeOutX + leftEyeInnerX)/2
			rightEyeY = (rightEyeOutY + rightEyeInnerY)/2
			rightEyeX = (rightEyeOutX + rightEyeInnerX)/2


			eyesCenterY = ((int)((leftEyeY + rightEyeY) /2))
			eyesCenterX = ((int)((leftEyeX + rightEyeX) /2))
			
			dy = (rightEyeY - leftEyeY)
			dx = (rightEyeX - leftEyeX)
			length = cv2.sqrt(dx*dx + dy*dy)
			angle = math.atan2(dy,dx) * 180.0/ np.pi
			scale = 1

			


#			rows, cols = image.shape[:2]
#			cols, rows = image.shape[:2]
			rows, cols = ((y1+h1), (x1+w1))
#			rows, cols = ((x1+w1), (y1+h1))
			
#			M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
#			M = cv2.getRotationMatrix2D((((int)eyesCenterX),((int)eyesCenterY)),angle,1)
			M = cv2.getRotationMatrix2D((eyesCenterX,eyesCenterY),angle,1)
#			image = cv2.warpAffine(image,M,(cols,rows))
#			image = cv2.warpAffine(image,M,(eyesCenterX*2,eyesCenterY*2))
			image = cv2.warpAffine(image,M,(desireWidth,desireHeight))
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			rects = detector(gray,1)
#			print("small image type:",image.dtype)
#			print("small gray image type:",gray.dtype)
			for(i, rect) in enumerate(rects):
				if i == 0:
					shape = predictor(gray, rect)
					shape = face_utils.shape_to_np(shape)
					(x,y,w,h) = face_utils.rect_to_bb(rect)
	#				cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
	#				for(x,y) in shape:
	#					cv2.circle(image,(x,y),1,(0,0,255),-1)
					(leftX,leftY) = shape[0]
					(rightX,rightY) = shape[16]
					(topX,topY) = shape[27]
					(bottomX, bottomY) = shape[57]
					finalWidth = (rightX - leftX)
					finalHeight = finalWidth
	#				centerFaceX_start = ((int)((leftX + rightX)/2 - finalWidth/2))
					centerFaceX_start = leftX
					centerFaceY_start = ((int)((topY + bottomY)/2 - finalHeight/2))
	#				centerFaceY_start = topY
#					print("current image shape :",image.shape)
#					print("crop image shape :", centerFaceY_start,finalHeight,centerFaceX_start,finalWidth)
					image = image[centerFaceY_start:centerFaceY_start+finalHeight,centerFaceX_start:centerFaceX_start + finalWidth]
					#width maybe zero
					try:
#						image = imutils.resize(image, width=112)
						if image.shape[0] == 0 or image.shape[1] == 0:
							image = None
						else:
							image = cv2.resize(image,(112,112),interpolation=cv2.INTER_CUBIC)
					except ZeroDivisionError:
						image = None
					#(width,height) = image.shape[:2]
					#if ((float)(width)) == 0 : 
					#else:
				
			
#切人脸的框。
#			image = image[y1:y1+h1,x1:x1+w1]
		if image is None:
			pass
		else:
			print("time count:",time_count)
			print("final image shape :",image.shape)
			
			#if out is None:
			#	(h,w) = image.shape[:2]
			#	out = cv2.VideoWriter("output.avi",fourcc, 20.0, (w ,h))
			#out.write(image)
			
#			cv2.imshow("frame",image)
			#0~100
#			cv2.imwrite("test"+str(time_count)+".jpg",[int(cv2.IMWRITE_JPEG_QUALITY),95],image)
			cv2.imwrite(str(time_count)+".jpg",image)
			if time_count == 982:
				break
			#0~9
			#cv2.imwrite("test"+str(time_count)+".png",[int(cv2.IMWRITE_PNG_COMPRESSION),5],image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

cap.release()
out.release()
cv2.destroyAllWindows()



