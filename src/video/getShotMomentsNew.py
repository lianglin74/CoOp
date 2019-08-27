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
import os.path

def setDebug(frame): 
    return frame > 2975 and frame < 3025
            
class Trajectory(object):
    def __init__(self, frameRate, debug):
        # parameter
        self.highRecall = True
        self.iouLowThresh = 0.01
        self.iouHighThresh = 0.55
        self.angleRimToBallThresh = 45.0
        self.ballAboveRimThresh = 0.0
        self.eventPadding = 1.0

        self.iouTime = -1
        self.frameRate = frameRate
        self.debug = debug
        self.clear()

    def add(self, ballRects, rimRects, frame):
        self.ballTraj.append(ballRects)
        self.rimTraj.append(rimRects)
        self.frameTraj.append(frame)
        # calculate IOU
        if objectExists(ballRects) and objectExists(rimRects):
            iou = maxIOAOnebb_in_list_of_bb(ballRects[0], rimRects)
        else:
            iou = 0.0

        self.iouTraj.append(iou)

        self.debug = setDebug(frame)

        if self.debug:
            print("Frame, iou, ballRects, rimRects: ", frame, iou, ballRects, rimRects)

        return iou

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
        self.iouTime = float(self.frameTraj[maxIouIndex]) / self.frameRate

        # add padding time
        # case WrongBasketball_1:
        endTime = max(endTime, self.iouTime + self.eventPadding)

        # to avoid small period (not good for demo show), adjust the starting time
        startTime = min(startTime, self.iouTime - self.eventPadding)
        startTime = max(0, startTime)

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

        return shot, startTime, endTime, eventType, self.iouTime, reason

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

    def clear(self):
        self.ballTraj = []
        self.rimTraj = []
        self.iouTraj = []
        self.frameTraj = []
        self.iouTime = -1.0

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

        # ---- Parameters ----
        self.basketBallConfThresh = 0.5
        self.rimConfThresh = 0.2
        self.backboardConfThresh = 0.1
        # When to start or end an event
        self.distanceBallToRimThresh = 3

    # instance method
    def findEvent(self):
        # Initialize
        eventStarted = False
        trajectory = Trajectory(self.frameRate, self.debug)
        prevRects = {"ball": [], "rim": [], "backboard": []}
        eventResults = []

        for row in tqdm(tsv_reader(self.odTSVFile)):
            self.imageCnt += 1

            self.debug = setDebug(self.imageCnt)

            if self.debug:
                print("image: ", self.imageCnt)
                print("second: ", self.imageCnt / self.frameRate)

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

            # Get the rim closest to ball
            rimRects = self.getClosestRimRects(ballRects, rimRects)

            #if self.debug:
            #    print("ballRects: ", ballRects)
            #    print("rimRects: ", rimRects)

            # ignore wrongly preditcted balls

            # Add missing ball if necessary
            # if eventStarted:
            #    rimBB, ballBB, backBoardBB = updateBallRects(rimBB, ballBB, backboardBB, trajectory)

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
            if eventStarted:
                iou = trajectory.add(ballRects, rimRects, self.imageCnt)
                
                eventEnded = self.checkWhetherEventEnded(ballRects, rimRects)
                if (eventEnded):
                    eventStarted = False
                    shot, startTime, endTime, eventType, iotTime, reason = trajectory.analyze()
                    if shot:
                        # update results
                        eventResults.append(
                            (startTime, endTime, eventType, iotTime, reason))
                    # else: #doing nothing
                    trajectory.clear()
                # else: #event going on, doing nothing
            else:
                eventStarted = self.checkWhetherEventStarted(
                    ballRects, rimRects)

        return eventResults

    def getClosestRimRects(self, ballRects, rimRects):
        if objectExists(ballRects) and objectExists(rimRects):            
            ballCenter = getCenterOfObject(ballRects[0])
            listOfBallToRimsDistance = [getDistanceOfTwoPoints(
                ballCenter, getCenterOfObject(b)) for b in rimRects]
            indexOfClosestRim, _ = minIndexVal(listOfBallToRimsDistance)
            rectClosestRim = rimRects[indexOfClosestRim]

            return [rectClosestRim]
        else:
            return rimRects

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

    def checkWhetherEventStarted(self, ballRects, rimRects):
        ballToRimDistance = self.getDistanceBalltoRim(ballRects, rimRects)
        widthRim = self.getWidthOfRim(rimRects)

        if ballToRimDistance is not None and widthRim is not None and ballToRimDistance < self.distanceBallToRimThresh * widthRim:
            return True
        else:
            return False

    #
    def checkWhetherEventEnded(self, ballRects, rimRects):
        ballToRimDistance = self.getDistanceBalltoRim(ballRects, rimRects)
        widthRim = self.getWidthOfRim(rimRects)

        if ballToRimDistance is not None and widthRim is not None and ballToRimDistance > self.distanceBallToRimThresh * widthRim:
            return True
        else:
            return False

    def getRimBallBackboard(self, rects):
        ballRects = []
        rimRects = []        
        backboardRects = []

        maxBasketBallConf = self.basketBallConfThresh
        maxBasketBallRectIndex = -1

        i = 0
        for r in rects:
            if r['class'] == 'basketball':
                if r['conf'] >= maxBasketBallConf:
                    maxBasketBallConf = r['conf']
                    maxBasketBallRectIndex = i
            elif r['class'] == 'basketball rim':
                if r['conf'] >= self.rimConfThresh:
                    rimRects.append(r)
            elif r['class'] == 'backboard':
                if r['conf'] >= self.backboardConfThresh:
                    backboardRects.append(r)
                    if self.debug and not objectExists(rimRects):
                        print("[Warning] image ",  self.imageCnt,
                              ": backboard found, but no rim")
            i += 1

        # get the ball with highest confidence score
        if maxBasketBallRectIndex != -1:
            ballRects.append(rects[maxBasketBallRectIndex])

        return ballRects, rimRects, backboardRects

    # if backboard shows, but rim does not show; add rim;
    def updateRimBackboardRects(self, rimRects, backboardRects, prevRects):
        if objectExists(backboardRects) and objectExists(prevRects["rim"]) and not objectExists(rimRects):
            rimRects.extend(prevRects["rim"])
        return rimRects, backboardRects

    def checkRimBackBoardCorrespondence(self):
        pass


def objectExists(objRectList):
    return len(objRectList) > 0


# Given two rectangles, rect0 is smaller. Check the ratio of intersection of rect0 and rect1 to area of rect0
def calculateIOA(rect0, rect1):
    '''
    x0, y1, x2, y3
    '''
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


def getPersonHoldingBall_loose(rimRect, ballRect, rects, debug):
    ballPersonIouThresh = 0
    rimPersonIouThresh = 0

    if debug:
        print("Rects:", rects)

    for r in rects:
        if r['class'] == 'person':
            personRect = r['rect']
            if debug:
                print("preson rect:", personRect)
                print("iouWithBall: ", calculate_iou(personRect, ballRect))
                print("iouWithPerson", calculate_iou(personRect, rimRect))

            if calculate_iou(personRect, ballRect) > ballPersonIouThresh and calculate_iou(personRect, rimRect) > rimPersonIouThresh:
                return True

    return False


def getPersonHoldingBall(rimRect, ballRect, rects, debug):
    ballPersonIouThresh = 0
    rimPersonIouThresh = 0

    for r in rects:
        if r['class'] == 'person':
            personRect = r['rect']

            if calculate_iou(personRect, ballRect) > ballPersonIouThresh and calculate_iou(personRect, rimRect) > rimPersonIouThresh \
                    and isAbove((personRect[0], personRect[1]), (rimRect[2], rimRect[3])) \
                    and getHeightOfRect(personRect) > 2.0 * getHeightOfRect(rimRect):
                return True
    return False


def toDegree(angle):
    return angle / math.pi * 180.0


def checkResultsOverlap(pred_results):
    l = len(pred_results)
    padding = 2
    i = 1

    newResults = []
    while i < l:
        v1 = pred_results[i-1]
        v2 = pred_results[i]

        if (v1[1] > v2[0] + padding):
            print("Found overlap pairs: ", v1, v2)
            exit()
        else:
            newResults.append(v1)

        i += 1

    if l >= 2:
        i = l-1
        v1 = pred_results[i-1]
        v2 = pred_results[i]
        if (v1[1] > v2[0] + padding):
            print("Found overlap pairs: ", v1, v2)
            exit()
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
        value for value in true_results if not findValueInPairList(value,  pred_results)]
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
    return any([v >= pair[0] and v <= pair[1] for v in true_results])


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


def getDistanceOfTwoPoints(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def maxIouOnebb_in_list_of_bb(bb, bbs):
    return max([calculate_iou(b['rect'], bb['rect']) for b in bbs])

# bb is supposed smaller than b in bbs. 
def maxIOAOnebb_in_list_of_bb(bb, bbs):
    return max([calculateIOA(bb['rect'], b['rect']) for b in bbs])

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


def writeToTSV(labelFile, pred_results):
    fileName = labelFile.replace(".tsv", "_events.tsv")
    videoFileName = labelFile.replace(".tsv", ".mp4")
    id = 0
    #className = "shot"
    padding = 0
    dictList = []
    for v in pred_results:
        id += 1
        value = {}

        value['id'] = id
        value['start'] = v[0] - padding
        value['end'] = v[1] + padding
        value['class'] = v[2]
        dictList.append(value)

    print('json format results:')
    print(dictList)

    f = open(fileName, 'w')
    f.write(videoFileName + '\t' + json.dumps(dictList) + '\n')
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
    treatingDunkAsShot = True
    labelCorrectionDict = {}
    
    while i < lp and j < lt:
        pRes = pred_results[i]
        tRes = true_results[j]
        if tRes[0] < pRes[0]:
            allTimePoints.append(tRes)
            y_pred.append( nonShotLabel )
            y_true.append( shotLabel if treatingDunkAsShot else tRes[1] )
            j += 1
        elif tRes[0] > pRes[1] + tolerance:
            allTimePoints.append(pRes)
            y_pred.append(  shotLabel if treatingDunkAsShot else pRes[2] )
            y_true.append( nonShotLabel )
            i += 1
        else:
            allTimePoints.append(tRes)            
            #assert(pRes[2] == tRes[1])
            y_pred.append(  shotLabel if treatingDunkAsShot else pRes[2] )
            y_true.append(  shotLabel if treatingDunkAsShot else tRes[1] )

            if correctLabel:
                print("predict, label: ", pRes[3], tRes[0])
                if (abs(tRes[0] - pRes[3]) > 2.5):
                    print("Label Correction failed! label, iouTime", tRes[0], pRes[3])
                else:
                    labelCorrectionDict[tRes[0]] = pRes[3]

            i += 1
            j += 1
    
    while i < lp:
        allTimePoints.append(pRes)
        y_pred.append( shotLabel if treatingDunkAsShot else pRes[2] )
        y_true.append( nonShotLabel )
        i += 1

    while j < lt:
        allTimePoints.append(tRes)        
        y_pred.append( nonShotLabel )
        y_true.append( shotLabel if treatingDunkAsShot else tRes[1] )
        j += 1

    return y_pred, y_true, allTimePoints, labelCorrectionDict

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

            y_pred, y_true, _, correctLabelsDict = getShotStats(pred_results, true_results)

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
    #main()
    #test_getShotStats()
    getValidationResults()
    #getTestingResults()
    # testGetDegreeOfTwoPoints()
    #test_getEventLabelsFromText()
    #getMiguTestingResults()
