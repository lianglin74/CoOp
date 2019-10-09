from __future__ import division
import cv2
import json
import os
from qd import tsv_io
from qd.tsv_io import tsv_reader
import sys
from scipy import constants
from video.labelViewerForVideo import drawLabel, getID2Labels
from video.generateTSVFromVideo import saveFramesByFrameList
from video.getShotMoments import getCenterOfObject, getFrameRate, getWidthOfObject, getHeightOfObject, getHeightOfRect
import copy
import math
import numbers

#formula: http://www.sohu.com/a/128231413_544219
class Parabola(object):
    def __init__(self, ballSize = 27.1, fps = 25.0):
        self.usingPixel = True
        self.ballSize = ballSize
        self.RealBallSize = 0.75 #meter
        self.ratio_pixel_to_meter = 1.0 / ballSize * self.RealBallSize
        self.ratio_frame_to_second = 1.0/fps
        self.constG_in_pixel = constants.g / self.ratio_pixel_to_meter
        self.constG = self.constG_in_pixel if self.usingPixel else constants.g

    # Given two positions of ball, calculate the initial horizontal velocity of the ball
    def calculateVx(self, p1, p2, dt):
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
    def calculateVy(self, p1, p2, dt):
        dy = p2[1] - p1[1] 
        return (dy/dt + self.constG* dt / 2.0)

    # dt2 = t2 - t1;
    # dt3 = t3 - t1; 
    # x = vx*t
    # y = vy*t - g*t^2/2
    def calculateP3(self, p1, p2, dt2, dt3):
        vx = self.calculateVx(p1, p2, dt2)
        vy = self.calculateVy(p1, p2, dt2)
        x = p1[0] + vx * dt3
        y = p1[1] + vy * dt3 - self.constG * dt3 * dt3 / 2.0
        return (x,y)

# ball falling with initial speed v0 (going down is the positive speed direction). 
class FreeFall(object):
    def __init__(self, ballSize = 27.1, fps = 25.0):
        self.usingPixel = False
        self.ballSize = ballSize
        self.RealBallSize = 0.75 #meter
        self.ratio_pixel_to_meter = 1.0 / ballSize * self.RealBallSize
        self.ratio_frame_to_second = 1.0/fps
        self.constG_in_pixel = constants.g / self.ratio_pixel_to_meter
        self.constG = self.constG_in_pixel if self.usingPixel else constants.g
    
    # Given two positions of ball, calculate the initial vertical velocity of the ball
    # Formula: 
    # v = v0 + g*t
    # y = v0 * t + 1/2*g*t^2    
    # ==>
    # y2 - y1  = v1*(t2 - t1) + 1/2*g(t2-t1)^2
    # dt = t2 - t1
    def calculateV1(self, y1, y2, dt):
        dy = y2 - y1
        return (dy/dt - self.constG* dt / 2.0)

    def calculateV0_mid(self, v1, y0, y1):
        dy1 = y1 - y0
        #print(v1*v1 - 2.0 * self.constG * dy1)
        tmp = v1*v1 - 2.0 * self.constG * dy1
        if tmp > 0:
            return math.sqrt(tmp)
        else:
            return -1
    
    def calculateV0_frame(self, y0, f1, y1, f2, y2):
        dt21 = (f2 - f1) * self.ratio_frame_to_second
        return self.calculateV0(y0, y1, y2, dt21)

    def calculateV0(self, y0, y1, y2, dt21):
        v1 = self.calculateV1(y1, y2, dt21)
        v0 = self.calculateV0_mid(v1, y0, y1)
        if not self.usingPixel:
            v0 *= self.ratio_pixel_to_meter
        return v0

    def test_calculateV0(self):
        # t0: # with ioa: 
        #f0 = 343
        ##rimRect_0 = [668.2710647583008, 251.72594833374023, 747.8637008666992, 317.29431533813477]
        #ballRect_0 = [725.5067958831787, 235.3461971282959, 754.8691806793213, 263.0349979400635]        
        #ballCenter = (740.18798828125, 249.1905975341797) 

        # t0: change to estimate ball position: 
        f0 = 348
        rimRect_0 = [688.5313129425049, 252.03049087524414, 747.4360942840576, 307.4753074645996]


        #t1:
        f1 = 351
        # ball center: 
        #p1 = (684.461669921875, 333.3879089355469)
        p1 = (684.461669921875, 326.3879089355469)
        ballRect_1 = [670.7639427185059, 318.2334566116333, 698.1593971252441, 348.54236125946045]
        rimRect_1 = [673.290449142456, 265.1431522369385, 727.8236865997314, 307.449987411499]
        
        # t2: 
        f2 = 352
        p2 = (684.1061401367188, 339.98974609375)
        ballRect_2 = [670.4249134063721, 326.41224193573, 697.7873668670654, 353.56725025177]
        rimRect_2 = [673.1895370483398, 258.88002586364746, 737.4427871704102, 306.8599033355713]

        # t3: 
        f3 = 355
        p3 = (666.480224609375, 395.86767578125)
        rimRect_3= [669.8128509521484, 249.37556648254395, 746.0129547119141, 307.29008293151855]

        # get rim bottom:
        rimBottom = (rimRect_1[3] + rimRect_2[3])/2.0
        print(rimBottom)
        heightOfBall = (getHeightOfRect(ballRect_1) + getHeightOfRect(ballRect_2))/2.0
        print(heightOfBall)
        # get y0:
        y0 = rimBottom - heightOfBall/2.0
        print('y0: ', y0)
        t_ratio = 1/25.0        

        t0 = f0 * t_ratio
        t1 = f1 *t_ratio
        y1 = p1[1]
        t2 = f2 *t_ratio
        y2 = p2[1]
        t3 = f3 * t_ratio
        y3 = p3[1]

        # using y1, y2: Not accurate
        print("--Rough f0: t1, t2", f0)
        self.verifyV0_simple(y0, t1, y1, t2, y2)

        # using y2, y3
        print("--Rough f0:t2, t3 ", f0)
        self.verifyV0_simple(y0, t2, y2, t3, y3)

        # using y1, y3
        print("--Rough f0: t1, t3", f0)
        self.verifyV0_simple(y0, t1, y1, t3, y3)

        print("--Real f1: t2, t3 ", f1)
        self.verifyV0_simple(y1, t2, y2, t3, y3)        

        
    def test_calculateV0_simple(self):
        t_shift = 10
        y_shift = 200

        t0 = 0
        v0 = 0
        y0 = 0 + y_shift

        t1 = 1 + t_shift
        v1 = self.constG
        y1 = 0.5*self.constG + y_shift
        
        t2 = 2 + t_shift
        v2 = 2*self.constG
        y2 = 2*self.constG + y_shift

        t3 = 3 + t_shift
        v3 = 3*self.constG
        y3 = 4.5*self.constG + y_shift

        self.verifyV0_simple(y0, t1, y1, t2, y2)
        self.verifyV0_simple(y0, t1, y1, t3, y3)

        self.verifyV0_simple(y1, t2, y2, t3, y3)

    def verifyV0_simple(self, y0, t1, y1, t2, y2, postProcess = True):
        dt21 = t2 - t1
        v0 = self.calculateV0( y0, y1, y2, dt21)
        v1 = self.calculateV1(y1,y2,dt21)
        v2 = v1 + self.constG * dt21
        dt10 = (v1 - v0) / self.constG
        cal_t0 = t1 - dt10

        print("v0: ", v0, "; v1: ", v1,  "; v2: ", v2,  "; dt10: ", dt10, "; cal_t0: ", cal_t0)

        if postProcess:
            print("speed to meter, time to frame")
            print("v0: ", v0 * self.ratio_pixel_to_meter , "; v1: ", v1 * self.ratio_pixel_to_meter,  "; v2: ", v2 * self.ratio_pixel_to_meter,  "; dt10: ", dt10 / self.ratio_frame_to_second, "; cal_t0: ", cal_t0 / self.ratio_frame_to_second)

        

# for frameID in frameList, using the first 2 as real; then estimate the others. 
def verifyBallPrediction(topDir, video_name, labelFileName, frameList, motionObj):
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
    print("labelIndexStartingFromZero: ", labelIndexStartingFromZero)

    p1Ball = None
    p2Ball = None
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

        print("i: ", i, "; frame: ", frameList[i])
        ballRect  = getRects(labels)       
        rimRect  = getRects(labels, "basketball rim")
        print("ball rect:", ballRect)
        print("rim rect: ", rimRect)
        
        if ballRect is not None:
            print("ball center: ", getCenterOfObject(getRects(labels)))
            p1Ball = p2Ball
            p2Ball = (ballRect, i)
        else:
            if p1Ball is not None and p2Ball is not None:
                p1 = getCenterOfObject(p1Ball[0])
                p2 = getCenterOfObject(p2Ball[0])            
                dt2 = (frameList[p2Ball[1]] - frameList[p1Ball[1]])/frameRate
                dt3 = (frameList[i] - frameList[p1Ball[1]])/frameRate
                newBallPosition = motionObj.calculateP3(p1, p2, dt2, dt3)
                print("NewBallPosition: ", newBallPosition)
            
            dummyBallRect = getDummyBall(p1Ball, p2Ball) if not usingNewCenter else getDummyBall(p1Ball, p2Ball, newBallPosition)
            labels.append(dummyBallRect)

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
    motionObj = Parabola()
    newBallPosition = motionObj.calculateP3(p1, p2, dt2, dt3)
    print("NewBallPosition: ", newBallPosition)
    print("real pos: ", p3)

def getRects(labels, className = 'basketball'):
    maxBasketBallConf = 0.1
    maxBasketRectIndex = -1
    ballRects = None
    
    i = 0
    for r in labels:
        if r['class'] == className:
            if r['conf'] >= maxBasketBallConf:
                maxBasketBallConf = r['conf']
                maxBasketRectIndex = i
        i += 1

    if maxBasketRectIndex != -1:
        ballRects = labels[maxBasketRectIndex]
    
    return ballRects

def getDummyBall(p1Ball, p2Ball, newCenter = None, debug=1):    
    if p1Ball is None or p2Ball is None:
        return None
    
    scaleFactor = 2.0

    widthOfBall = getWidthOfObject(p2Ball[0]) * scaleFactor
    heightOfBall = getHeightOfObject(p2Ball[0]) * scaleFactor
    if debug:
        print('widthOfBall: ', widthOfBall)
        print('heightOfBall: ', heightOfBall)

    x1, y1 = getCenterOfObject(p1Ball[0])
    x2, y2 = getCenterOfObject(p2Ball[0])

    if newCenter is None:
        x = 2 * x2 - x1
        y = 2 * y2 - y1
    else:
        x = newCenter[0]
        y = newCenter[1]

    dummyBallObj = copy.deepcopy(p2Ball[0])
    dummyBallObj['rect'][0] = x - (widthOfBall + 1.0) / 2
    dummyBallObj['rect'][1] = y - (heightOfBall + 1.0) / 2
    dummyBallObj['rect'][2] = x + (widthOfBall + 1.0) / 2
    dummyBallObj['rect'][3] = y + (heightOfBall + 1.0) / 2

    if debug:
        print('dummyBallObj: ', dummyBallObj)

    return dummyBallObj

def test_verifyBallPrediction_1():
    topdir = "/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/validation/extracted/"

    video_name = "1552493137730_sc99_01_q2.mp4"
    labelFileName = "1552493137730_sc99_01_q2.tsv"
    
    #frameList = [319, 320, 342]
    frameList = list(range(2368, 2378+1))

    motionObj = Parabola()
    verifyBallPrediction(topdir, video_name, labelFileName, frameList, motionObj)

def test_verifyBallPrediction():
    topdir = "/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/validation/extracted/"

    video_name = "1551538896210_sc99_01_q1.mp4"
    labelFileName = "1551538896210_sc99_01_q1.tsv"
    
    #frameList = [319, 320, 342]
    frameList = list(range(341, 364+1))

    motionObj = Parabola()
    verifyBallPrediction(topdir, video_name, labelFileName, frameList, motionObj)

def test_FreeFall():
    motionObj = FreeFall()

    #motionObj.test_calculateV0_simple()
    motionObj.test_calculateV0()

if __name__ == '__main__':
    #test_verifyBallPrediction()
    #test_getPredictedPosition()

    test_FreeFall()
