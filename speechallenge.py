import cv2 as cv
import numpy as np
import random


#Getting speed from the text file
speed = []
with open("data/train.txt") as f:
    for line in f:
        speed.append(f.readline())

speed = [x.strip() for x in speed]

iter_speed = iter(speed) #making the speed list an iterable

#retrieving all frames from video and saving them in all_frames
#all_frames[np.ndarry] = speed for that particular frame
cap = cv.VideoCapture("data/train.mp4")
all_frames = []
ret, frame1 = cap.read()
while ret:
    temp = []
    temp.append(frame1)
    temp.append(float(next(iter_speed)))
    all_frames.append(temp)
    ret, frame1 = cap.read()




#Dense Optical Flow based on the Gunner Farneback's algorithm
cap = cv.VideoCapture("data/train.mp4")
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.imshow('frame2',bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',frame2)
        cv.imwrite('opticalhsv.png',bgr)
    prvs = next
cap.release()
cv.destroyAllWindows()
