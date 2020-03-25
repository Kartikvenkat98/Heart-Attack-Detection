import cv2
import sys
import os
import shutil

def FrameCapture(path):

	dir_path = "./frames/"

	if os.path.exists(dir_path):
		shutil.rmtree(dir_path)

	os.mkdir(dir_path)
	  
	vidObj = cv2.VideoCapture(path) 
  
	count = 0

	success, image = vidObj.read()
  
	while success: 

		cv2.imwrite(dir_path + "/frame%d.jpg" % count, image) 
		success, image = vidObj.read()
		count += 1
		print(count)


FrameCapture(sys.argv[1])