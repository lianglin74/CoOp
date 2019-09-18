import cv2
import json
from qd import tsv_io
from qd.process_tsv import onebb_in_list_of_bb
from qd.tsv_io import tsv_reader, tsv_writer
from qd.qd_common import calculate_iou

from video.getShotMoments import f1Report,getEventLabelsFromText
from video.getEventLabels import getVideoAndEventLabels, labelConverter

from tqdm import tqdm
import numpy as np
import math
import copy
import os

def setDebug(frame): 
    return False
    #return frame > 2370 and frame <2387
    #return frame > 300 and frame < 525

class Trajectory(object):
    def __init__(self, frameRate, debug, videoFile):
        # Important parameters to tune:
        self.highRecall = True
        self.iouLowThresh = 0.01
        self.iouHighThresh = 0.55
        self.shotDetectWindow = 2.0 #at least >=2.0
        self.largestEventWindow = 4.0
        self.wideEventWindow = True
        self.angleRimToBallThresh = 45.0
        self.ballAboveRimThresh = 0.0
        self.eventPadding = 1.0
        # for dunk
        self.rimPersonIoaThresh = 0.05
        self.personHeightToRimRatio = 2.0
        self.dunkTimeWindow = 0.25
        
        # To solve the problem in case: Case "RimNotGood_1"
        self.enlargeRatio = 1.5

        # for debugging purpose
        self.printMissingBallFrames = False
        self.debug = debug

        # initialization
        self.ioaTime = -1
        self.frameRate = frameRate
        self.videoFile = videoFile
        
        self.clear()

    def add(self, ballRects, rimRects, personRects, frame):
        self.ballTraj.append(ballRects)
        self.rimTraj.append(rimRects)
        self.personTraj.append(personRects)
        self.frameTraj.append(frame)
        # calculate IOA between ball and rim (for shot detection)
        if objectExists(ballRects) and objectExists(rimRects):
            iou = maxIOAOnebb_in_list_of_bb(ballRects[0], rimRects, self.enlargeRatio)
        else:
            iou = 0.0

        self.iouTraj.append(iou)

        self.debug = setDebug(frame)

        if self.debug:
            print("Frame, iou, ballRects, rimRects: ", frame, iou, ballRects, rimRects)

        return iou

    def getDunkFrame(self):
        #dunkFrameList = []
        for rimRects, personRects, frame in zip(self.rimTraj, self.personTraj, self.frameTraj):
            if objectExists(rimRects) and objectExists(personRects):
                personRect = personRects[0]['rect']
                rimRect = rimRects[0]['rect']
                if getHeightOfRect(personRect) > self.personHeightToRimRatio * getHeightOfRect(rimRect) \
                  and isAbove((personRect[0], personRect[1]), (rimRect[0], rimRect[1])) \
                  and calculateIOA(rimRect, personRect) > self.rimPersonIoaThresh:                    
                    return True, frame
        
        return False, 0

    def analyze(self):
        # filtering wrong object detection

        # interpolate missing balls

        # If no IOU, then return False
        shot = False
        startTime = float(self.frameTraj[0]) / self.frameRate
        endTime = float(self.frameTraj[-1]) / self.frameRate
        eventType = "shot"
        reason = "N/A"

        # check iou
        maxIouIndex, maxIouValue = maxIndexVal(self.iouTraj)
        self.ioaTime = float(self.frameTraj[maxIouIndex]) / self.frameRate

        # add padding time
        # case WrongBasketball_1:
        if self.wideEventWindow: 
            endTime = max(endTime, self.ioaTime + self.eventPadding)
        else:
            endTime = min(endTime, self.ioaTime + self.eventPadding)

        # to avoid small period (not good for demo show), adjust the starting time
        if self.wideEventWindow:
            startTime = min(startTime, self.ioaTime - self.eventPadding)
        else:
            startTime = max(startTime, self.ioaTime - self.eventPadding)
            
        startTime = max(0, startTime)

        # Eliminate wrongly extra big window
        if endTime - startTime > self.largestEventWindow:
            endTime = min(endTime, self.ioaTime + self.largestEventWindow / 2.0)
            startTime = startTime = max(startTime, self.ioaTime - self.largestEventWindow / 2.0)

        if maxIouValue > self.iouHighThresh:
            shot = True
            reason = 'highIou'
        elif maxIouValue > self.iouLowThresh:
            # check the necessary condition of shot: ball is lower than rim and within a cone
            if self.necessaryCondition(maxIouIndex):
                if self.highRecall:                
                    shot = True
                    reason = 'HighRecall'
                else:
                    if self.extraConditions(maxIouIndex):
                        shot = True
                        reason = 'extraCond'

        # get the most likely dunk person (for dunk detection)
        ret, dunkFrame = self.getDunkFrame()
        if ret:
            dunkTime = dunkFrame / self.frameRate
            print("Finding a possible dunk at frame: ", dunkFrame, "; time: ", dunkTime)            
            if abs(dunkTime - self.ioaTime) < self.dunkTimeWindow:
                eventType = "dunk"

        # output the frames of the first missing ball
        if self.printMissingBallFrames and shot:
            self.writeMissingBallFrames()

        return shot, startTime, endTime, eventType, self.ioaTime, reason

    def writeMissingBallFrames(self):
        fileName = self.videoFile.replace(".mp4", "-missingBallFrames.txt")
        with open(fileName, 'a') as f:
            for i, ballRects in enumerate(self.ballTraj):
                if i > 0 and objectExists(ballRects) and objectExists(self.ballTraj[i  - 1]) and ballRects[0]['conf'] == self.ballTraj[i  - 1][0]['conf']:
                    f.write(str(self.frameTraj[i]) + "\t" + str(self.ballTraj[i  - 1][0]['rect']) + "\n")
                    f.close()
                    print("Missing ball at frame: ", self.frameTraj[i], "Time: ", self.frameTraj[i]/self.frameRate, "Previous frame: rect: ", self.ballTraj[i  - 1][0]['rect'], 'conf: ', ballRects[0]['conf'])
                    return

    def extraCondition(self, maxIouIndex):
        l = len(self.ballTraj)

        ballIndex = self.findFirstBallPositionLowerThanRim(maxIouIndex)
        if ballIndex == -1:
            return False
        else:
            ballRects = self.ballTraj[ballIndex]
            rimRects = self.rimTraj[ballIndex]
            ballCenter = getCenterOfObject(ballRects[0])
            centerOfRim = getCenterOfObject(rimRects[0])
            angleRimToBall = getDegreeOfTwoPoints(centerOfRim, ballCenter)
            if abs(angleRimToBall - 90) < self.angleRimToBallThresh:
                return True
            else:
                return False

    def necessaryCondition(self, maxIouIndex):
        return True

    def findFirstBallPositionLowerThanRim(self, maxIouIndex):
        i = maxIouIndex
        l = len(self.ballTraj)
        while i < l:
            if objectExists(self.ballTraj[i]) and objectExists(self.rimTraj[i]) and not self.ballAboveRim(self.ballTraj[i], self.rimTraj[i]):
                return i
            i += 1
        return -1
    
    def findLastBallPositionLowerThanRim(self, maxIouIndex):
        l = len(self.ballTraj)
        i = l - 1
        pos = i
        while i > 0:
            if objectExists(self.ballTraj[i]) and objectExists(self.rimTraj[i]) and not self.ballAboveRim(self.ballTraj[i], self.rimTraj[i]):
                pos = i
            else:
                break
            i -= 1
        return -1

    def guessAllBallPositions(self):
        # find first iou and last iou index
        ioaIndices = [ i for i, ioa in enumerate(self.iouTraj) if ioa > self.iouLowThresh ]
        

    def filterWrongBallDetection(self):
        # Using idea similar to median filtering
        pass;

    def clear(self):
        self.ballTraj = []
        self.rimTraj = []
        self.personTraj = []
        self.iouTraj = []
        self.frameTraj = []
        self.ioaTime = -1.0

    def ballAboveRim(self, ballRects, rimRects):
        if objectExists(ballRects) and objectExists(rimRects):
            ballCenter = getCenterOfObject(ballRects[0])
            centerOfRim = getCenterOfObject(rimRects[0])
            return isAbove(ballCenter, centerOfRim, self.ballAboveRimThresh)
        else:
            return False

    def predictBallRects(self, debug=0):
        prevBallObjs = self.ballTraj
        l = len(prevBallObjs)
        if l < 2 or not objectExists(prevBallObjs[l-1]) or not objectExists(prevBallObjs[l-2]):
            return []

        widthOfBall = getWidthOfObject(prevBallObjs[l-1][0])
        heightOfBall = getHeightOfObject(prevBallObjs[l-1][0])
        if debug:
            print('widthOfBall: ', widthOfBall)
            print('heightOfBall: ', heightOfBall)

        x1, y1 = getCenterOfObject(prevBallObjs[l-2][0])
        x2, y2 = getCenterOfObject(prevBallObjs[l-1][0])

        x = 2 * x2 - x1
        y = 2 * y2 - y1

        dummyBallObj = copy.deepcopy(prevBallObjs[l-1][0])
        dummyBallObj['rect'][0] = x - (widthOfBall + 1.0) / 2
        dummyBallObj['rect'][1] = y - (heightOfBall + 1.0) / 2
        dummyBallObj['rect'][2] = x + (widthOfBall + 1.0) / 2
        dummyBallObj['rect'][3] = y + (heightOfBall + 1.0) / 2

        if debug:
            print('dummyBallObj: ', dummyBallObj)

        return [dummyBallObj]

def maxIndexVal(values):
    max_index, max_value = max(enumerate(values), key=lambda p: p[1])
    return max_index, max_value


def minIndexVal(values):
    min_index, min_value = min(enumerate(values), key=lambda p: p[1])
    return min_index, min_value


class EventDetector(object):
    # Initializer / Instance attributes
    def __init__(self, odTSVFile, videoFile):
        # ---- Parameters ----
        self.basketBallConfThresh = 0.5
        self.rimConfThresh = 0.1
          #Case "WrongBasketball_2": needs rimConfThresh to be below 0.19. 
          #Case "RimNotGood_1": prefer rimConfThresh to be above 0.35

        self.backboardConfThresh = 0.1
        # When to start or end an event
        self.distanceBallToRimThresh = 3

        # for dunk: 
        self.ballPersonIouThresh = 0.2
        self.personThresh = 0.2

        # initialize
        self.odTSVFile = odTSVFile
        self.videoFile = videoFile

        if not os.path.exists(odTSVFile):
            print("The following file does not exists! Exiting!\n", odTSVFile)
            exit()
        if not os.path.exists(videoFile):
            print("The following file does not exists! Exiting!\n", videoFile)
            exit()

        #self.eventResults = []
        self.debug = False
        self.imageCnt = 0

        self.frameRate = getFrameRate(videoFile)

    # instance method
    def findEvent(self):
        # Initialize
        eventStarted = False
        startTime = 0
        endTimeRes = -1
        
        trajectory = Trajectory(self.frameRate, self.debug, self.videoFile)
        prevRects = {"ball": [], "rim": [], "backboard": []}
        eventResults = []

        for row in tqdm(tsv_reader(self.odTSVFile)):
            curTime = self.imageCnt / self.frameRate
            
            self.debug = setDebug(self.imageCnt)

            if self.debug:
                print("image: ", self.imageCnt)
                print("second: ", curTime)

            #key = row[0]
            rects = json.loads(row[1])

            # Check whether there is rim and ball; backboard
            #   Note: ballRects is a list of rect object, i.e., ballRects[0] would be the rect for ball if the list is not empty
            ballRects, rimRects, backboardRects = self.getRimBallBackboard(rects)

            # Add missing rim or backboard always
            rimRects, backboardRects = self.updateRimBackboardRects(
                rimRects, backboardRects, prevRects)

            # Add missing ball: for case: BallNotDetected_1:
            if not objectExists(ballRects):
                #print("Before adding ball rects:", ballRects)
                ballRects = trajectory.predictBallRects()
                #print("After adding ball rects:", ballRects)

            # Get the ball, rim pair with smallest distance if there are multiple balls or rims
            ballRects, rimRects = self.getClosestBallRimPair(ballRects, rimRects)

            assert(len(ballRects) <= 1)
            assert(len(rimRects) <= 1)

            # ignore wrongly preditcted balls

            # get the persons holding ball
            personRects = self.getPersonHoldingBall(ballRects, rects, debug = self.debug)

            # Store the prev rects
            prevRects["ball"] = ballRects
            prevRects["rim"] = rimRects
            prevRects["backboard"] = backboardRects

            # Check the relative positision of rim and ball
            #distanceBallToRim, ballAboveRim = getRelativePosition(rimBB, ballBB)

            # Update the event status: started or not (if started, record the trajectory; else, clear up)
            #   If distance < thresh for two or three frames (filtering out wrong labels), then started.
            #   If eventStarted and distance > thresh for two or three frames (), then an event found.
            #   Analyze the trajectory.
            #   Add the results if a shot is found.

            if not eventStarted:                
                eventStarted = self.checkWhetherEventStarted(endTimeRes, curTime, ballRects, rimRects)
                if eventStarted:
                    startTime = curTime
                    if self.debug:
                        print("Event started!")
            else:
                iou = trajectory.add(ballRects, rimRects, personRects, self.imageCnt)
                
                eventEnded = self.checkWhetherEventEnded(ballRects, rimRects, curTime, startTime, trajectory.shotDetectWindow)
                if (eventEnded):
                    if self.debug:
                        print("Event ended!")
                        
                    eventStarted = False
                    shot, startTimeRes, endTimeRes, eventType, ioaTime, reason = trajectory.analyze()
                    if shot:
                        # update results
                        eventResults.append(
                            (startTimeRes, endTimeRes, eventType, ioaTime, reason))
                    # else: #doing nothing
                    trajectory.clear()
                # else: #event going on, doing nothing

                
            self.imageCnt += 1

        return eventResults

    def getClosestBallRimPair(self, ballRects, rimRects):
        cntBall = len(ballRects)
        cntRim = len(rimRects)        
        if cntBall >= 1 and cntRim >= 1:
            ballRect, rimRect = getClosestRects(ballRects, rimRects)
            return [ballRect], [rimRect]
        elif cntBall >= 1 and cntRim < 1:
            return [getRectWithHighestConfScore(ballRects)], rimRects
        elif cntBall < 1 and cntRim >= 1:
            return ballRects, [getRectWithHighestConfScore(rimRects)]
        else: # cntBall < 1 and cntRim < 1
            return ballRects, rimRects

    def getWidthOfRim(self, rimRects):
        rimExists = objectExists(rimRects)

        if rimExists:
            rectClosestRim = rimRects[0]
            #print("rectClosestRim", rectClosestRim)
            return getWidthOfObject(rectClosestRim)
        else:
            return None

    def getDistanceBalltoRim(self, ballRects, rimRects):
        ballExists = objectExists(ballRects)
        rimExists = objectExists(rimRects)

        if ballExists and rimExists:            
            rectClosestRim = rimRects[0]
            ballCenter = getCenterOfObject(ballRects[0])
            centerOfRim = getCenterOfObject(rectClosestRim)
            return getDistanceOfTwoPoints(ballCenter, centerOfRim)
        else:
            return None

    def checkWhetherEventStarted(self, endTime, curTime, ballRects, rimRects):
        # To prevent wrong detection of ball and then start another event too early. 
        # Case "WrongBasketball_2". 
        if (curTime < endTime):
            return False; 

        ballToRimDistance = self.getDistanceBalltoRim(ballRects, rimRects)
        widthRim = self.getWidthOfRim(rimRects)

        if ballToRimDistance is not None and widthRim is not None and ballToRimDistance < self.distanceBallToRimThresh * widthRim:
            return True
        else:
            return False

    #
    def checkWhetherEventEnded(self, ballRects, rimRects, curTime, startTime, shotDetectWindow):
        if self.debug:
            print("CurTime ", curTime)
            print("startTime", startTime)
            
        # To handle the false case where event ends earlier because of false ball detection
        # Example: case "WrongBallDetected_3"
        if curTime < startTime + shotDetectWindow:
            return False

        ballToRimDistance = self.getDistanceBalltoRim(ballRects, rimRects)
        widthRim = self.getWidthOfRim(rimRects)

        if self.debug:
            print("ball Rim Distance: ", ballToRimDistance)
            print("Width of rim", widthRim)

        if ballToRimDistance is not None and widthRim is not None and ballToRimDistance > self.distanceBallToRimThresh * widthRim:
            return True
        else:
            return False

    def getRimBallBackboard(self, rects):
        ballRects = []
        rimRects = []        
        backboardRects = []

        for r in rects:
            if r['class'] == 'basketball':
                if r['conf'] >= self.basketBallConfThresh:
                    ballRects.append(r)
            elif r['class'] == 'basketball rim':
                if r['conf'] >= self.rimConfThresh:
                    rimRects.append(r)
            elif r['class'] == 'backboard':
                if r['conf'] >= self.backboardConfThresh:
                    backboardRects.append(r)
        
        if self.debug and objectExists(backboardRects) and not objectExists(rimRects):
            print("[Warning] image ",  self.imageCnt, ": backboard found, but no rim")

        return ballRects, rimRects, backboardRects

    # if backboard shows, but rim does not show; add rim;
    def updateRimBackboardRects(self, rimRects, backboardRects, prevRects):
        if objectExists(backboardRects) and objectExists(prevRects["rim"]) and not objectExists(rimRects):
            rimRects.extend(prevRects["rim"])
        return rimRects, backboardRects

    def checkRimBackBoardCorrespondence(self):
        pass

    def getPersonHoldingBall(self, ballRects, rects, debug = 0):
        if not objectExists(ballRects):
            return []

        #sort objects by area
        rects.sort(key = lambda instance: areaOfRect(instance['rect']), reverse = True)

        for r in rects:
            if r['class'] == 'person':
                if r['conf'] < self.personThresh:
                    continue

                personRect = r['rect']

                if calculateIOA(ballRects[0]['rect'], personRect) > self.ballPersonIouThresh:
                    if debug:                 
                        print("Frame:", self.imageCnt, "; Finding a person holding ball:", r)
                    return [r]
        return []

def objectExists(objRectList):
    return len(objRectList) > 0

def areaOfRect(rect):
    return (rect[2] - rect[0]) * (rect[3] - rect[1])

def enlargeRect(rect0, enlargeRatio):
    widthOfBall = getWidthOfRect(rect0) * enlargeRatio
    heightOfBall = getHeightOfRect(rect0) * enlargeRatio

    x, y = getCenterOfRect(rect0)
    
    rect0[0] = x - (widthOfBall ) / 2
    rect0[1] = y - (heightOfBall ) / 2
    rect0[2] = x + (widthOfBall ) / 2
    rect0[3] = y + (heightOfBall ) / 2

    return rect0

def getRectWithHighestConfScore(rects):
    return max(rects, key = lambda rect: rect['conf'])

def test_getRectWithHighestConfScore():
    rects = [{'rect': [90, 90, 110, 110], 'class': 'basketball', 'conf': 0.7321000099182129, 'obj': 0.7321000099182129}, 
            {'rect': [190, 190, 210, 210], 'class': 'basketball', 'conf': 0.85, 'obj': 0.7321000099182129}]
    print(getRectWithHighestConfScore(rects))

def getClosestRects(ballRects, rimRects):
    distanceList = [ [i, j, getTwoRectCenterDistance(ballRect, rimRect)] for i, ballRect in enumerate(ballRects) for j, rimRect in enumerate(rimRects)]
    minEle = min(distanceList, key = lambda ele : ele[2])
    return ballRects[minEle[0]], rimRects[minEle[1]]

def test_getClosestRects():
    ballRects = [{'rect': [90, 90, 110, 110], 'class': 'basketball', 'conf': 0.7321000099182129, 'obj': 0.7321000099182129}, 
        {'rect': [190, 190, 210, 210], 'class': 'basketball', 'conf': 0.65, 'obj': 0.7321000099182129}]
        #center: 100, 100; 200, 200
    rimRects = [{'rect': [30, 30, 70, 70], 'class': 'basketball rim', 'conf': 0.7321000099182129, 'obj': 0.7321000099182129}, 
        {'rect': [160, 160, 200, 200], 'class': 'basketball', 'conf': 0.65, 'obj': 0.7321000099182129}]
        #center: 50, 50; 180, 180
    print(getClosestRects(ballRects, rimRects))

# Given two rectangles, rect0 is smaller. Check the ratio of intersection of rect0 and rect1 to area of rect0
def calculateIOA(rect0, rect1, enlargeRatio = 1.0):
    '''
    x0, y1, x2, y3
    '''
    if enlargeRatio > 1.0:
        rect0 = enlargeRect(rect0, enlargeRatio)

    w = min(rect0[2], rect1[2]) - max(rect0[0], rect1[0])
    if w < 0:
        return 0
    h = min(rect0[3], rect1[3]) - max(rect0[1], rect1[1])
    if h < 0:
        return 0
    intersection = w * h
    
    #a1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    a0 = (rect0[2] - rect0[0]) * (rect0[3] - rect0[1])

    # a0 is not zero
    assert (a0);
    
    return intersection / a0


def getPersonHoldingBallOverlapWithRim(rimRect, ballRect, rects, debug = 0):
    ballPersonIouThresh = 0
    rimPersonIouThresh = 0
    personThresh = 0.2

    #sort objects by area
    rects.sort(key = lambda instance: areaOfRect(instance['rect']), reverse = True)

    for r in rects:
        if r['class'] == 'person':
            if r['conf'] < personThresh:
                continue

            personRect = r['rect']

            if getHeightOfRect(personRect) > 2.0 * getHeightOfRect(rimRect) \
                  and isAbove((personRect[0], personRect[1]), (rimRect[0], rimRect[1])) \
                  and calculate_iou(personRect, ballRect) > ballPersonIouThresh and calculate_iou(personRect, rimRect) > rimPersonIouThresh:                    
                return [r]
    return []


def toDegree(angle):
    return angle / math.pi * 180.0


def checkResultsOverlap(pred_results):
    l = len(pred_results)
    padding = 1
    i = 1

    newResults = []
    while i < l:
        v1 = pred_results[i-1]
        v2 = pred_results[i]

        if (v1[1] > v2[0] + padding):
            print("Found overlap pairs: ", v1, v2)
            #exit()
        else:
            newResults.append(v1)

        i += 1

    if l >= 2:
        i = l-1
        v1 = pred_results[i-1]
        v2 = pred_results[i]
        if (v1[1] > v2[0] + padding):
            print("Found overlap pairs: ", v1, v2)
            #exit()
        else:
            newResults.append(v2)

    return newResults

    if l == 1:
        return pred_results


def calculateF1(pred_results, true_results):
    print("pred_results: ", pred_results, len(pred_results))
    print("true_results: ", true_results, len(true_results))

    truePositiveList = [
        value for value in pred_results if findPairInValueList(value, true_results)]
    print("TruePositve: ", truePositiveList)
    # trueNegative =
    falsePositiveList = [
        value for value in pred_results if not findPairInValueList(value, true_results)]
    print("FlasePositve: ", falsePositiveList)
    falseNegativeList = [
        value for value in true_results if not findValueInPairList(value[0],  pred_results)]
    print("falseNegative: ", falseNegativeList)

    precision = len(truePositiveList) / (len(truePositiveList) +
                                         len(falsePositiveList) + 0.0) if len(truePositiveList) else 0

    recall = len(truePositiveList) / (len(truePositiveList) +
                                      len(falseNegativeList) + 0.0) if len(truePositiveList) else 0

    F1 = 2.0 / (1.0/precision + 1.0/recall)

    print("precision: ", precision)
    print("recall: ", recall)
    print("F1: ", F1)


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def findPairInValueList(pair, true_results):
    return any([v[0] >= pair[0] and v[0] <= pair[1] for v in true_results])


def findValueInPairList(v, pred_results):
    return any([v >= pair[0] and v <= pair[1] for pair in pred_results])


def isAbove(p1, p2, thresh=0.0):
    return p1[1] < p2[1] - thresh


def getWidthOfObject(bb):
    x1 = bb['rect'][0]
    x2 = bb['rect'][2]

    return x2 - x1


def getHeightOfObject(bb):
    y1 = bb['rect'][1]
    y2 = bb['rect'][3]

    return y2 - y1

def getWidthOfRect(rect):
    x1 = rect[0]
    x2 = rect[2]

    return x2 - x1

def getHeightOfRect(rect):
    y1 = rect[1]
    y2 = rect[3]

    return y2 - y1

def getCenterOfObject(bb):
    x1 = bb['rect'][0]
    y1 = bb['rect'][1]
    x2 = bb['rect'][2]
    y2 = bb['rect'][3]
    xc = x1 + (x2 - x1) / 2.0
    yc = y1 + (y2 - y1) / 2.0
    return (xc, yc)

def getTwoRectCenterDistance(ballRect, rimRect):
    ballCenter = getCenterOfObject(ballRect)
    rimCenter =  getCenterOfObject(rimRect)
    return getDistanceOfTwoPoints(ballCenter, rimCenter)

def getCenterOfRect(rect):
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]
    xc = x1 + (x2 - x1) / 2.0
    yc = y1 + (y2 - y1) / 2.0
    return (xc, yc)

def getDistanceOfTwoPoints(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def maxIouOnebb_in_list_of_bb(bb, bbs):
    return max([calculate_iou(b['rect'], bb['rect']) for b in bbs])

# bb is supposed smaller than b in bbs. 
def maxIOAOnebb_in_list_of_bb(bb, bbs, enlargeRatio = 1.0):
    return max([calculateIOA(bb['rect'], b['rect'], enlargeRatio) for b in bbs])

# from p1 pointing to p2. Good to use


def getDegreeOfTwoPoints(p1, p2):
    return getAngleOfTwoPoints(p1, p2) / math.pi * 180.0


def getAngleOfTwoPoints(p1, p2):
    deltaX = -(p1[0] - p2[0])
    len = np.linalg.norm(np.array(p1) - np.array(p2))
    orgAngle = np.arccos(deltaX / len)
    if (p1[1] > p2[1]):
        orgAngle = math.pi * 2.0 - orgAngle
    return orgAngle


def test_getAngleOfTwoPoints():
    rimP2 = (1, 1)

    listOfPoints = [[0, 0], [0, 1], [1, 2], [2, 2],
                    [2, 1], [1, 0], [2, 0], [0, 2]]

    expectedResults = [45, 0, 270, 225,
                       180, 90, 135, 315]

    eps = 1e-6

    for i, v in enumerate(listOfPoints):
        p1 = v
        degree = getAngleOfTwoPoints(p1, rimP2) / math.pi * 180
        if abs(degree - expectedResults[i]) > eps:
            print("Error founded. Testing failed")
            print("p1 is: ", p1)
            print("Angle is: ", )
        else:
            print("Testing passed")


def getFrameRate(video_name):
    cap = cv2.VideoCapture(video_name)

    # get the total frame count
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using OLD version", fps)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using new version", fps)

    cap.release()
    return fps


def writeToTSV(labelFile, results, timePoint = False):
    fileName = labelFile.replace(".tsv", "_events.tsv")
    videoFileName = labelFile.replace(".tsv", ".mp4")
    
    dictList = buildEventLists(results, timePoint, rounding = True)

    f = open(fileName, 'w')
    f.write(videoFileName + '\t' + json.dumps(dictList) + '\n')
    f.close()

def buildEventLists(results, timePoint, rounding = False):
    id = 0
    #className = "shot"
    frontPadding = 1.2 if timePoint else 0 
    endPadding = 0.8 if timePoint else 0
    dictList = []
    for v in results:
        id += 1
        value = {}

        value['id'] = id
        value['start'] = v[0] - frontPadding if timePoint else  v[0] - frontPadding
        if (value['start'] < 0):
            value['start'] = 0

        value['end'] = v[0] + endPadding if timePoint else v[1] + endPadding
        if rounding:
            value['start'] = int(value['start'])
            value['end'] = math.ceil(value['end'])
        value['class'] = v[1] if timePoint else v[2]
        dictList.append(value)

    print('json format results:')
    print(dictList)
    return dictList

def writeTrainingLabelsForAutoML(labelFile, results, timePoint = True, suffix = None):
    ''' if suffix is not None: 
        fileName = labelFile.replace(".tsv", "_autoML_" + suffix + ".csv")
    else:
        fileName = labelFile.replace(".tsv", "_autoML.csv") '''

    videoFileName = labelFile.replace(".tsv", ".mp4")

    videoFileBase = os.path.basename(videoFileName)
    videoFilePath = os.path.dirname(videoFileName)

    if "data/video/CBA/CBA_demo_v3" in videoFilePath:
        prefix = "gs://yaoguang-central-storage/shotDunk/"
        videoFileNamePrefix = videoFileBase[:-4]
    elif "data/video/CBA/CBA_5_test_videos/test/extracted" in videoFilePath:
        prefix = "gs://yaoguang-central-storage/shotDunk/test/"
        videoFileNamePrefix = videoFileBase[:-7]
    else:        
        prefix = "gs://yaoguang-central-storage/shotDunk/tmp/"
        videoFileNamePrefix = videoFileBase[:-7]
    
    testList = ['647b025243d74e719b36d24a0c19df37_sc99', # 39 shot, 1 dunk; 
        'CBA1' #33 shots, 11 dunks;
        'NBA1' #16 shots, 0 dunk
        ]
    # vadalition: 10 dunk; test: 8 dunk; 
    
    if videoFileNamePrefix in testList:
        fileName = "/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/shotDetectionAutoML/autoGen/test.csv"
    else:
        fileName = "/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/shotDetectionAutoML/autoGen/train.csv"

    print("----writing to:", fileName)
    print("prefix: ", prefix)
    print("events: ", results)

    dictList = buildEventLists(results, timePoint)

    f = open(fileName, 'a')
    
    for value in dictList:
        f.write(','.join([prefix + os.path.basename(videoFileName), value['class'], '%.3f' % value['start'], '%.3f' % value['end']]) + "\n")
    f.close()

def read_file_to_list(file_name):
    res_lists = []
    with open(file_name, 'r') as file:  # Use file to refer to the file object
        data = file.read()
        res_lists = data.split()

    return res_lists


''' def getShotAPI(videoFileName, predict_file):
    frameRate = getFrameRate(videoFileName)
    pred_results = findShot(predict_file, frameRate)

    writeToTSV(predict_file, pred_results)

    return pred_results '''

def getShotStats(pred_results, true_results):
    #pred_results: [(s1, e1, class1), (s2, e2, class2), ...]
    #true_results: [(t1, class1), (t2, class2)]
    lp = len(pred_results)
    lt = len(true_results)
    print("pred_results: ", pred_results, lp)
    print("true_results: ", true_results, lt)

    nonShotLabel = "nonShot"
    shotLabel = "shot"
    y_pred = []
    y_true = []

    i = 0
    j = 0
    allTimePoints = []

    tolerance = 0.5
    correctLabel = False
    treatingDunkAsShot = False
    labelCorrectionDict = {}
    
    while i < lp and j < lt:
        pRes = pred_results[i]
        tRes = true_results[j]
        if tRes[0] < pRes[0]:
            allTimePoints.append(tRes)
            y_pred.append( (nonShotLabel, None) )
            y_true.append( (shotLabel if treatingDunkAsShot else tRes[1], tRes) )
            j += 1
        elif tRes[0] > pRes[1] + tolerance:
            allTimePoints.append(pRes)
            y_pred.append( (shotLabel if treatingDunkAsShot else pRes[2], pRes) )
            y_true.append( (nonShotLabel, None) )
            i += 1
        else:
            allTimePoints.append(tRes)
            #assert(pRes[2] == tRes[1])
            y_pred.append( (shotLabel if treatingDunkAsShot else pRes[2], pRes) )
            y_true.append( (shotLabel if treatingDunkAsShot else tRes[1], tRes) )

            if correctLabel:
                print("predict, label: ", pRes[3], tRes[0])
                if (abs(tRes[0] - pRes[3]) > tolerance):
                    print("Label Correction failed! label, ioaTime", tRes[0], pRes[3])
                else:
                    labelCorrectionDict[tRes[0]] = pRes[3]

            i += 1
            j += 1
    
    while i < lp:
        pRes = pred_results[i]
        allTimePoints.append(pRes)
        y_pred.append( (shotLabel if treatingDunkAsShot else pRes[2], pRes) )
        y_true.append( (nonShotLabel, None) )
        i += 1

    while j < lt:
        tRes = true_results[j]
        allTimePoints.append(tRes)        
        y_pred.append( (nonShotLabel, None) )
        y_true.append( (shotLabel if treatingDunkAsShot else tRes[1], tRes) )
        j += 1

    return y_pred, y_true, allTimePoints, labelCorrectionDict

def confusionMatrixReport(y_pred, y_pred_pointer, y_true, y_true_pointer):
    # Reference: https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python
    '''Computes a confusion matrix using numpy for two np.arrays
    true and pred.

    Results are identical (and similar in computation time) to:
        "from sklearn.metrics import confusion_matrix"

    However, this function avoids the dependency on sklearn.'''
    classes = np.unique(y_true + y_pred).tolist()

    true = [ classes.index(v) for v in y_true]
    pred = [ classes.index(v) for v in y_pred]

    K = len(classes) # Number of classes 
    result = np.zeros((K, K))
    
    cfMatrix = np.zeros((K, K)).tolist()
    for i in range(K):
        for j in range(K):
            cfMatrix[i][j] = []

    for i in range(len(y_true)):
        result[true[i]][pred[i]] += 1
        cfMatrix[true[i]][pred[i]].append((y_true_pointer[i], y_pred_pointer[i]))

    # print the count:
    print("Detailed confusion matrix: ")
    print("Classes: ", classes)
    #print("y_predict: ", y_pred)
    #print("y_true: ", y_true)
    #print("y_predict_pointer: ", y_pred_pointer)
    #print("y_true_pointer: ", y_true_pointer)
    print("Every row is for true result, while every column is for the prediction")
    print(result)
    #print(cfMatrix)
    print("\n----Wrong predictions:")
    for i in range(K):
        for j in range(K):
            if i != j and len(cfMatrix[i][j]) > 0:
                print("--", i, j, ": predict ", classes[i], " as ", classes[j], ": ", len(cfMatrix[i][j]), ";  ".join(str(x) for x in cfMatrix[i][j]))
    
    print("\n----Correct predictions:")
    for i in range(K):
        for j in range(K):
            if i == j:
                print("--", i, j, ": predict ", classes[i], " as ", classes[j], ": ", len(cfMatrix[i][j]), ";  ".join(str(x) for x in cfMatrix[i][j]))

    print("\n")
    return cfMatrix

def getFalsePositiveRes(allTimePoints):
    eventLength = 2.0
    padding = 1.0

    falsePositiveRes = []
    l = len(allTimePoints)
    prevTimePoint = -10.0
    for i,v in enumerate(allTimePoints):
        nextTimePoint = 3600*1000.0      
        if len(v) > 2:
            if i < l - 1:
                next = allTimePoints[i + 1]
                nextTimePoint = next[0] if len(next) == 2 else next[3]
            if v[3] - prevTimePoint > eventLength + padding and nextTimePoint - v[3] > eventLength + padding:
                falsePositiveRes.append((v[3], 'None_of_the_above'))
                prevTimePoint = v[3]
        else:
            prevTimePoint = v[0]
    
    return falsePositiveRes

def getNegativeRes(allTimePoints):
    eventLength = 2.0
    padding = 2.0

    negativeRes = []
    tpList = [ v[0] if len(v) == 2 else v[3] for v in allTimePoints]


    l = len(tpList)
    i = 1
    while i < l:
        gap = (tpList[i] - tpList[i - 1])
        if gap > 2 * eventLength + padding:
            negativeRes.append(((tpList[i - 1] + tpList[i])/2.0, "None_of_the_above"))
        i += 1

    return negativeRes

def main():
    dir = "/mnt/gpu02_raid/data/video/CBA/CBA_demo_v2/"
    labelFiles = "labellist.txt"

    labelFileList = read_file_to_list(dir + labelFiles)
    #predict_file = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_chop/TSV/head350_prediction_1551538896210_sc99_01_q1.tsv"
    #predict_file = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_chop/prediction_1551538896210_sc99_01_q1.tsv"

    # Hack!!
    videoFile = "/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/validation/extracted/1551538896210_sc99_01_q1.mp4"

    for predict_file in labelFileList:
        print("----Processing file: ", predict_file)
        #pred_results = findShot(dir + predict_file)

        eventDetector = EventDetector(dir + predict_file, videoFile)
        pred_results = eventDetector.findEvent()

        checkResultsOverlap(pred_results)

        true_results = None
        # if predict_file == "prediction_1551538896210_sc99_01_q1.tsv":
        if predict_file == "1551538896210_sc99_01_q1_pd.tsv":
            tmpRes = [(13.922896, 'shot'), (36.051405, 'shot'), (55.823736, 'shot'), (120.206053, 'shot'), (151.343382, 'shot'), (158.211654, 'shot'), (186.743713, 'shot'), (328.851485, 'shot'), (344.177011, 'shot'), (350.888363, 'shot'), (386.284587, 'dunk/layup'), (443.872138, 'shot'), (469.697181, 'shot'), (479.717574, 'shot'), (526.101491, 'shot'), (586.654381, 'shot')]
            true_results = [ v[0] for v in tmpRes ]

        if predict_file == "1552493137730_sc99_01_q1_pd.tsv":
            true_results = [81, 95, 110, 135, 148, 152, 282, 298]

        if predict_file == "NBA1.tsv":
            true_results = [float(v)
                            for v in read_file_to_list(dir + "NBA1.gt.txt")]

        if predict_file == "CBA1.tsv":
            true_results = [float(v)
                            for v in read_file_to_list(dir + "CBA1.gt.txt")]

        if predict_file == "CBA2.tsv":
            true_results = [float(v)
                            for v in read_file_to_list(dir + "CBA2.gt.txt")]

        if predict_file == "NBA2.tsv":
            true_results = [float(v)
                            for v in read_file_to_list(dir + "NBA2.gt.txt")]

        if true_results is not None:
            print("----calculate F1 for file: ", predict_file)
            print("True_results:", true_results)
            calculateF1(pred_results, true_results)

        writeToTSV(predict_file, pred_results)

def calculateF1andWriteRes(dir, odFileList, eventLabelJsonFile, textLabels = False, textLabelFolder = None):
    # Used to write labels for GL autoML training
    writeAutoMLLabel = True

    odFileList = read_file_to_list(dir + odFileList)
    #predict_file = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_chop/TSV/head350_prediction_1551538896210_sc99_01_q1.tsv"
    #predict_file = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_chop/prediction_1551538896210_sc99_01_q1.tsv"

    overallPred = []
    overallTrue = []
    allReports = ""

    allCorrectLabels = {}

    for odFileName in odFileList:        
        predict_file = dir + odFileName
        videoFileName = odFileName.replace("tsv", "mp4")

        print("----Processing file: ", predict_file)
        eventDetector = EventDetector(predict_file, dir + videoFileName)
        pred_results = eventDetector.findEvent()
        #pred_results = findShot(predict_file)

        checkResultsOverlap(pred_results)
        
        ret = True
        if textLabels:
            true_results = getEventLabelsFromText(textLabelFolder + odFileName.replace('tsv', 'GTevents.txt'))
        else:
            ret, true_results = getVideoAndEventLabels(eventLabelJsonFile, videoFileName)
        if ret: 
            allReports += "--Report for file: " + videoFileName + "\n"

            print("----calculate F1 for file: ", predict_file)            
            #print("True_results:", true_results)
            #calculateF1(pred_results, true_results)

            y_pred_combo, y_true_combo, allTimePoints, correctLabelsDict = getShotStats(pred_results, true_results)
            #print(y_pred_combo)
            #print(y_true_combo)

            y_pred, y_pred_pointer = zip(*y_pred_combo)
            y_true, y_true_pointer = zip(*y_true_combo)

            confusionMatrixReport(y_pred, y_pred_pointer, y_true, y_true_pointer)

            if writeAutoMLLabel: 
                falsePositiveRes = getFalsePositiveRes(allTimePoints)
                #print("False Positive Results: ", falsePositiveRes)
                negativeRes = getNegativeRes(allTimePoints)
            
            allCorrectLabels[videoFileName] = correctLabelsDict

            overallPred.extend(y_pred)
            overallTrue.extend(y_true)

            allReports += f1Report(y_pred, y_true)
        else:
            print("!! cannot find label for video file: ", videoFileName)
            exit()

        # write events
        writeToTSV(predict_file, pred_results)
        #writeToTSV(predict_file, true_results, True)
        if writeAutoMLLabel:
            writeTrainingLabelsForAutoML(predict_file, true_results, timePoint = True)
            writeTrainingLabelsForAutoML(predict_file, falsePositiveRes, timePoint = True, suffix = "fakeShot")
            writeTrainingLabelsForAutoML(predict_file, negativeRes, timePoint = True, suffix = "nonShot")

    print("====F1 report for all the data: ")
    f1Report(overallPred, overallTrue)
    print(allReports)
    print(allCorrectLabels)

def getValidationResults():
    dir = "/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/validation/extracted/"    
    odFileList = "odFilelist.txt"
    eventLabelJsonFile = '/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/test/extracted/label/Project_all_corrected_manual.aucvl'
    calculateF1andWriteRes(dir, odFileList, eventLabelJsonFile)

def getTestingResults():
    dir = "/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/test/extracted/"    
    odFileList = "odFilelist.txt"
    eventLabelJsonFile = '/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/test/extracted/label/Project_all_corrected_manual.aucvl'
    calculateF1andWriteRes(dir, odFileList, eventLabelJsonFile)

def getMiguTestingResults():
    dir = "/mnt/gpu02_raid/data/video/CBA/CBA_demo_v3/"
    odFileList = "odFilelist.txt"
    textFileFolder = '/mnt/gpu02_raid/data/video/CBA/CBA_demo_v3/shotDunkLabels/'
    calculateF1andWriteRes(dir, odFileList, '', textLabels = True, textLabelFolder = textFileFolder)


if __name__ == '__main__':
    getValidationResults()
    getTestingResults()
    getMiguTestingResults()
    
    #main()
    #test_getShotStats()
    # testGetDegreeOfTwoPoints()
    #test_getEventLabelsFromText()    
    #test_getClosestRects()
    #test_getRectWithHighestConfScore()
