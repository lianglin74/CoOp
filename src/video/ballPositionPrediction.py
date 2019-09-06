from __future__ import division
import cv2
import json
import os
from qd import tsv_io
from qd.tsv_io import tsv_reader
import sys
from scipy.constants import gravitational_constant
from video.labelViewerForVideo import drawLabel, getID2Labels
from video.generateTSVFromVideo import saveFramesByFrameList
from video.getShotMoments import getCenterOfObject, getFrameRate, getWidthOfObject, getHeightOfObject
import copy

#formula: http://www.sohu.com/a/128231413_544219

# Given two positions of ball, calculate the initial horizontal velocity of the ball
def calculateVx(p1, p2, dt):
    return (p2[0] - p1[0]) / dt

# Given two positions of ball, calculate the initial vertical velocity of the ball
# Formula: 
# y = x vy/vx - g * x^2 / (2 * vx ^2)
# Assume (x1, y1) as the origin where ball is thrown with vx, vy
# (y - y1) = (x - x1) * vy/vx - g * (x - x1) ^2 / (2 * vx ^2)
# ==>
# vy = (dy + g * dx^2/(2vx^2) ) *vx / dx
# vy = (dy + g * dt^2/2) / dt
# vy = (dy/dt + g*dt/2)
def calculateVy(p1, p2, dt):
    dy = p2[1] - p1[1] 
    return (dy/dt + gravitational_constant* dt / 2.0)

# dt2 = t2 - t1;
# dt3 = t3 - t1; 
# x = vx*t
# y = vy*t - g*t^2/2
def calculateP3(p1, p2, dt2, dt3):
    vx = calculateVx(p1, p2, dt2)
    vy = calculateVy(p1, p2, dt2)
    x = p1[0] + vx * dt3
    y = p1[1] + vy * dt3 - gravitational_constant * dt3 * dt3 / 2.0
    return (x,y)


# for frameID in frameList, using the first 2 as real; then estimate the others. 
def verifyBallPrediction(topDir, video_name, labelFileName, frameList):
    l = len(frameList)
    #get 
    imageList = saveFramesByFrameList(topDir, video_name, frameList, returnFrames = 1)

    # get frame rate
    frameRate = getFrameRate(topDir + "/" + video_name)

    #read all labels
    id2Labels = getID2Labels(topDir + "/" + labelFileName)
    print("len(id2Labels)", len(id2Labels))
    
    sepSign = "$"

    labelIndexStartingFromZero = video_name + sepSign + '0' in id2Labels

    prevBallObjs=[]
    usingNewCenter = 1
    i = 0
    while i < l and i >= 0:    
        frame = imageList[i]
        imageId = video_name + sepSign + str(frameList[i] if labelIndexStartingFromZero else frameList[i] + 1)

        # get labels
        if imageId not in id2Labels:
            exit()
        else:
            labels = id2Labels[imageId]

        print("i: ", i)
        ballRect  = getBallRects(labels)        
        print("ball rect:", ballRect)
        if ballRect is not None:
            print("ball center: ", getCenterOfObject(getBallRects(labels)))
            prevBallObjs.append(ballRect)

        if i == 0:
            p1 = getCenterOfObject(ballRect)

        if i == 1:            
            p2 = getCenterOfObject(ballRect)
            
        if i > 1:
            dt2 = (frameList[1] - frameList[0])/frameRate
            dt3 = (frameList[i] - frameList[0])/frameRate
            newBallPosition = calculateP3(p1, p2, dt2, dt3)
            print("NewBallPosition: ", newBallPosition)            
            
            if ballRect is None:                
                dummyBallRect = getDummyBall(prevBallObjs) if not usingNewCenter else getDummyBall(prevBallObjs, newBallPosition)
                labels.append(dummyBallRect)
                prevBallObjs.append(dummyBallRect)

            #frame = cv2.putText(frame, '*', newBallPosition, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        frame = drawLabel(frame, labels)
        text = imageId
        frame = cv2.putText(frame, text, (int(0), int(60)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("withBox", frame)

        k = cv2.waitKeyEx(0)
        if k == 65361 or k == 2424832:  # left
            i -= 1
        elif k == 65363 or k == 32 or k == 2555904:  # right or space
            i += 1
        elif k == 27 or k == 113:  # esc
            exit()    

def test_getPredictedPosition():
    dt2 = 1.0/25
    dt3 = 2.0/25
    
    p1 = (1302.039306640625, 193.6150360107422)
    p2 = (1278.2122802734375, 174.70913696289062)
    p3 = (759.9295654296875, 221.84617614746094)
    newBallPosition = calculateP3(p1, p2, dt2, dt3)
    print("NewBallPosition: ", newBallPosition)
    print("real pos: ", p3)

def getBallRects(labels):        
    maxBasketBallConf = 0.1
    maxBasketRectIndex = -1
    ballRects = None
    
    i = 0
    for r in labels:
        if r['class'] == 'basketball':
            if r['conf'] >= maxBasketBallConf:
                maxBasketBallConf = r['conf']
                maxBasketRectIndex = i
        i += 1

    if maxBasketRectIndex != -1:
        ballRects = labels[maxBasketRectIndex]
    
    return ballRects


def getDummyBall(prevBallObjs, newCenter = None, debug=1):
    l = len(prevBallObjs)
    if l < 2:
        return None

    scaleFactor = 2.0

    widthOfBall = getWidthOfObject(prevBallObjs[l-1]) * scaleFactor
    heightOfBall = getHeightOfObject(prevBallObjs[l-1]) * scaleFactor
    if debug:
        print('widthOfBall: ', widthOfBall)
        print('heightOfBall: ', heightOfBall)

    x1, y1 = getCenterOfObject(prevBallObjs[l-2])
    x2, y2 = getCenterOfObject(prevBallObjs[l-1])

    if newCenter is None:
        x = 2 * x2 - x1
        y = 2 * y2 - y1
    else:
        x = newCenter[0]
        y = newCenter[1]

    dummyBallObj = copy.deepcopy(prevBallObjs[l-1])
    dummyBallObj['rect'][0] = x - (widthOfBall + 1.0) / 2
    dummyBallObj['rect'][1] = y - (heightOfBall + 1.0) / 2
    dummyBallObj['rect'][2] = x + (widthOfBall + 1.0) / 2
    dummyBallObj['rect'][3] = y + (heightOfBall + 1.0) / 2

    if debug:
        print('dummyBallObj: ', dummyBallObj)

    return dummyBallObj

def test_verifyBallPrediction():
    topdir = "/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/validation/extracted/"

    video_name = "1552493137730_sc99_01_q2.mp4"
    labelFileName = "1552493137730_sc99_01_q2.tsv"
    
    #frameList = [319, 320, 342]
    frameList = list(range(2368, 2378+1))

    verifyBallPrediction(topdir, video_name, labelFileName, frameList)

if __name__ == '__main__':
    test_verifyBallPrediction()
    #test_getPredictedPosition()