import cv2

def FrameCapture(path): 
      
    vidObj = cv2.VideoCapture(path) 
  
    count = 0

    success, image = vidObj.read()
  
    while success: 

        cv2.imwrite("./frames/frame%d.jpg" % count, image) 
        success, image = vidObj.read()
        count += 1
        print(count)

if __name__ == '__main__': 
   
    FrameCapture("./videos/video_1.mp4")