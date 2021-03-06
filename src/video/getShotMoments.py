import cv2
import json
from qd import tsv_io
from qd.process_tsv import onebb_in_list_of_bb
from qd.tsv_io import tsv_reader, tsv_writer
from qd.qd_common import calculate_iou

from tqdm import tqdm
import numpy as np
import math
import copy
from video.getEventLabels import getVideoAndEventLabels, labelConverter
from sklearn.metrics import classification_report,confusion_matrix
import os


def findShot(predict_file, frameRate=25.0):
    # parameters:
    addMissingRim = True
    addMissingBall = True

    className = "shot"

    basketBallThresh = 0.5
    rimThresh = 0.2
    backBoardThresh = 0.1
    rimBallIouLowerThresh = 0.01
    rimBallIouHigherThresh = 0.20

    angleRimToBallThresh = 30.0/180*math.pi

    # the time period between two shots
    oneShotTimethresh = 2

    # 2.0 * rim width
    distanceFromBallToRimToTrack = 3
    angleThresh = 120.0/180*math.pi
    #angleThreshBelowRim = 30.0/180*math.pi

    debug = 0

    bufferSeconds = 2
    totalBufferingFrames = 50
    # buffer for 3 seconds?
    pred_key_to_rects = {}

    pred_results = []
    iou_list = []
    ball_above_rim = []

    pred_results_angle = []

    imageCnt = 0

    upperBallLocation = (-1, -1)
    #startToTrackBall = F
    eventStart = False
    #eventEnd = False
    angleBallToRim = 270/180.0*math.pi
    #padding = 1.25
    padding = 1.75

    startTime = -1
    endTime = -1

    iouTime = -10
    personTime = -10
    prevShotTime = -10

    largeDistanceCount = 0
    LargeBallRimDistanceCountThresh = 5
    timeSinceEventStart = 0

    ballAboveRimThresh = 0

    prevRimObj = None
    prevBallObjs = []

    for row in tqdm(tsv_reader(predict_file)):
        imageCnt += 1

        # if (imageCnt > 9645 and imageCnt < 9650):
        if (imageCnt > 8200 and imageCnt < 8218):
            debug = 0

        if debug:
            print("image: ", imageCnt)
            print("second: ", imageCnt / frameRate)

        key = row[0]
        rects = json.loads(row[1])

        rimExists = False
        ballExists = False
        backboardExists = False
        maxBasketBallConf = basketBallThresh
        maxBasketRectIndex = -1
        ballRects = None
        rimRects = []

        i = 0
        for r in rects:
            if r['class'] == 'basketball':
                if r['conf'] >= maxBasketBallConf:
                    maxBasketBallConf = r['conf']
                    maxBasketRectIndex = i
                    ballExists = True
            elif r['class'] == 'basketball rim':
                if r['conf'] >= rimThresh:
                    rimRects.append(r)
                    rimExists = True
            elif r['class'] == 'backboard':
                if r['conf'] >= backBoardThresh:
                    # filteredRects.append(r)
                    backboardExists = True
                    if debug and not rimExists:
                        print("[Warning] image ",  imageCnt,
                              ": backboard found, but no rim")
            i += 1

        if addMissingRim and not rimExists and backboardExists and prevRimObj is not None:
            rimRects.append(prevRimObj)
            rimExists = True
            if debug:
                print("Adding a rim rect: ", rimRects)

        if maxBasketRectIndex != -1:
            ballRects = rects[maxBasketRectIndex]

        if debug and imageCnt == 344:
            print("Rects: ", rects)
            print("rimExists: ", rimExists)
            print("ballExists: ", ballExists)

        if not ballExists and addMissingBall and eventStart and iouTime < startTime and len(prevBallObjs) >= 2:
            ballRects = predictBallRects(prevBallObjs, debug)
            ballExists = True
            if debug:
                print("predict ball rects as: ", ballRects)

        if rimExists and ballExists:
            assert ballRects['class'] == "basketball"

            iou = maxIouOnebb_in_list_of_bb(ballRects, rimRects)

            # use ball rim angle to filter out
            ballCenter = getCenterOfObject(ballRects)

            listOfBallToRimsDistance = [getDistanceOfTwoPoints(
                ballCenter, getCenterOfObject(b)) for b in rimRects]

            distanceFromBallToClosetRim = min(listOfBallToRimsDistance)
            indexOfClosestRim = listOfBallToRimsDistance.index(
                distanceFromBallToClosetRim)
            rectClosestRim = rimRects[indexOfClosestRim]
            prevRimObj = rectClosestRim
            widthRim = getWidthOfObject(rectClosestRim)
            centerOfRim = getCenterOfObject(rectClosestRim)

            #ballAboveRimThresh = widthRim

            if iou > rimBallIouLowerThresh:
                currentTime = imageCnt/frameRate
                iouTime = currentTime
                if 0:
                    print("--Processing image: ", imageCnt,
                          "; second: ", currentTime)
                    print("Found both rim and ball with iou: ", iou)
                    print("Ball is above rim: ", isAbove(
                        ballCenter, centerOfRim, ballAboveRimThresh))
                if len(pred_results) == 0 or currentTime - pred_results[-1][1] > oneShotTimethresh:
                    pred_results.append(
                        (currentTime - oneShotTimethresh/2.0, currentTime + oneShotTimethresh/2.0, 'shot'))
                    iou_list.append(iou)
                    ball_above_rim.append(
                        isAbove(ballCenter, centerOfRim, ballAboveRimThresh))

                if (iou > rimBallIouHigherThresh):
                    print("!!Found a shot")

            if debug:
                print("ballCenter: ", ballCenter)

            if debug:
                print("ballRects:", ballRects)
                print("rimRect:", rimRects)
                print("iou:", iou)

            if debug:
                print("eventStart: ", eventStart)
                print("distanceFromBallToClosetRim: ",
                      distanceFromBallToClosetRim)
                print("distance Thresh:", distanceFromBallToRimToTrack * widthRim)
                print("centerOfRim: ", centerOfRim)

            # start to check people
            if (getPersonHoldingBall(rectClosestRim['rect'], ballRects['rect'], rects, debug)):
                if debug:
                    print("Found a person holding ball: ", imageCnt)
                personTime = imageCnt/frameRate

            if addMissingBall and distanceFromBallToClosetRim < distanceFromBallToRimToTrack * widthRim and isAbove(ballCenter, centerOfRim, ballAboveRimThresh):
                prevBallObjs.append(ballRects)
            else:
                prevBallObjs = []

            # start to check angle
            if not eventStart and imageCnt/frameRate > prevShotTime + padding:
                if distanceFromBallToClosetRim < distanceFromBallToRimToTrack * widthRim and isAbove(ballCenter, centerOfRim, ballAboveRimThresh):
                    angleBallToRim = getAngleOfTwoPoints(
                        ballCenter, centerOfRim)
                    eventStart = True
                    timeSinceEventStart = 0
                    startTime = imageCnt/frameRate
                    if debug:
                        print("---key event start")
                        print("angleBallToRim: ", toDegree(angleBallToRim))
            elif eventStart:  # event started
                timeSinceEventStart += 1.0 / frameRate

                if timeSinceEventStart >= oneShotTimethresh * 1:
                    eventStart = False
                    timeSinceEventStart = 0

                if distanceFromBallToClosetRim <= distanceFromBallToRimToTrack * widthRim and not isAbove(ballCenter, centerOfRim, ballAboveRimThresh):
                    angleRimToBall = getAngleOfTwoPoints(
                        centerOfRim, ballCenter)

                    if debug:
                        print("realAngle: ", toDegree(angleRimToBall))
                        print("relative angle: ", toDegree(
                            abs(angleRimToBall - angleBallToRim)))

                    if angleRimToBall > angleRimToBallThresh:
                        falsePositive = True

                    endTime = imageCnt/frameRate
                    if (abs(angleRimToBall - angleBallToRim) < angleThresh) and iouTime > startTime - padding and iouTime < endTime + padding:
                        print("Start time: ", startTime,
                              "; endtime: ", endTime)
                        print("Finding one shot by angle analysis: ",
                              (imageCnt/frameRate))
                        className = "shot"
                        prevShotTime = endTime
                        if personTime > startTime - padding and personTime < endTime + padding:
                            print("Finding one DUNK shot by angle analysis: ",
                                  (imageCnt/frameRate))
                            className = "dunk"
                        print("Adding result: ", (max(startTime -
                                                      padding, 0.0), endTime + padding, className))
                        pred_results_angle.append(
                            (max(startTime - padding, 0.0), endTime + padding, className, iouTime))
                        eventStart = False
                        timeSinceEventStart = 0
                    else:  # not a shot
                        if debug:
                            print("Not a shot")
                else:
                    if debug:
                        print("Skipping")
            # end

    print(pred_results)
    print(iou_list)
    print(ball_above_rim)
    print(pred_results_angle)

    return pred_results_angle
    # return pred_results


def predictBallRects(prevBallObjs, debug=0):
    l = len(prevBallObjs)
    if l < 2:
        return None

    widthOfBall = getWidthOfObject(prevBallObjs[l-1])
    heightOfBall = getHeightOfObject(prevBallObjs[l-1])
    if debug:
        print('widthOfBall: ', widthOfBall)
        print('heightOfBall: ', heightOfBall)

    x1, y1 = getCenterOfObject(prevBallObjs[l-2])
    x2, y2 = getCenterOfObject(prevBallObjs[l-1])

    x = 2 * x2 - x1
    y = 2 * y2 - y1

    dummyBallObj = copy.deepcopy(prevBallObjs[l-1])
    dummyBallObj['rect'][0] = x - (widthOfBall + 1.0) / 2
    dummyBallObj['rect'][1] = y - (heightOfBall + 1.0) / 2
    dummyBallObj['rect'][2] = x + (widthOfBall + 1.0) / 2
    dummyBallObj['rect'][3] = y + (heightOfBall + 1.0) / 2

    if debug:
        print('dummyBallObj: ', dummyBallObj)

    return dummyBallObj


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
                
                if debug:
                    print("preson rect:", r)
                    print("Rim rect: ", rimRect)
                    print("iouWithBall: ", calculate_iou(personRect, ballRect))
                    print("iouWithPerson", calculate_iou(personRect, rimRect))
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

def f1Report(y_pred, y_true, printDetails = False, allTimePoints = None):
    if printDetails:
        print("allPoints: \n", allTimePoints)
        print("Prediction: \n", y_pred)
        print("Truth: \n", y_true)
        
    #print(classification_report(y_true, y_pred, target_names = classes, digits = 4))    
    #print("Confusion matrix:\n%s" % confusion_matrix(y_true, y_pred, labels = classes))
    f1ReportText = classification_report(y_true, y_pred, digits = 4)
    cmText = "Confusion matrix:\n%s\n" % confusion_matrix(y_true, y_pred)
    print(f1ReportText)
    print(cmText)
    return f1ReportText + cmText

def test_getShotStats():
    pred_results = [(13.56, 14.44, 'shot', 'extraCond'), (35.76, 36.04, 'shot', 'extraCond'), (134.28, 135.96, 'shot', 'extraCond'), (157.28, 158.36, 'shot', 'extraCond'), (186.32, 187.16, 'shot', 'extraCond'), (198.24, 200.88, 'shot', 'extraCond'), (208.08, 208.64, 'shot', 'extraCond'), (328.24, 329.48, 'shot', 'extraCond'), (344.0, 344.08, 'shot', 'extraCond'), (344.16, 344.2, 'shot', 'extraCond'), (350.4, 351.32, 'shot', 'extraCond'), (385.6, 386.12, 'shot', 'extraCond'), (386.24, 386.92, 'shot', 'extraCond'), (408.92, 409.04, 'shot', 'extraCond'), (442.36, 445.92, 'shot', 'extraCond'), (478.92, 480.56, 'shot', 'extraCond'), (586.2, 586.96, 'shot', 'extraCond')]
    true_results = [(13.922896, 'shot'), (36.051405, 'shot'), (55.823736, 'shot'), (120.206053, 'shot'), (151.343382, 'shot'), (158.211654, 'shot'), (186.743713, 'shot'), (328.851485, 'shot'), (344.177011, 'shot'), (350.888363, 'shot'), (386.284587, 'dunk/layup'), (443.872138, 'shot'), (469.697181, 'shot'), (479.717574, 'shot'), (526.101491, 'shot'), (586.654381, 'shot')]
    true_results = [ (v[0], 'shot') for v in true_results]    

    y_pred, y_true, allTimePoints, _ = getShotStats(pred_results, true_results)
    f1Report(y_pred, y_true, True, allTimePoints)
    #expected F1 score:
    #precision:  0.5882352941176471
    #recall:  0.625
    #F1:  0.6060606060606061    
    
    #expected confusion matrix output:
    #Confusion matrix:
    #[[ 0  7]
    # [ 6 10]]


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

# from p1 pointing to p2. Good to use


def getAngleOfTwoPoints(p1, p2):
    deltaX = -(p1[0] - p2[0])
    len = np.linalg.norm(np.array(p1) - np.array(p2))
    orgAngle = np.arccos(deltaX / len)
    if (p1[1] > p2[1]):
        orgAngle = math.pi * 2.0 - orgAngle
    return orgAngle


def maxIouOnebb_in_list_of_bb(bb, bbs):
    return max([calculate_iou(b['rect'], bb['rect']) for b in bbs])


def testGetDegreeOfTwoPoints_v1():
    listOfPoints = [[0, 0], [0, 1], [1, 2], [2, 2], [2, 1], [1, 0]]
    for v in listOfPoints:
        p1 = np.array(v)
        p2 = np.array((1, 1))
        print("p1 is: ", p1)
        print("Angle is: ", angle_between(p1, p2) / math.pi * 180)


def testGetDegreeOfTwoPoints():
    listOfPoints = [[0, 0], [0, 1], [1, 2], [
        2, 2], [2, 1], [1, 0], [2, 0], [0, 2]]
    for v in listOfPoints:
        p1 = v

        # rim
        p2 = (1, 1)
        print("p1 is: ", p1)
        print("Angle is: ", getAngleOfTwoPoints(p1, p2) / math.pi * 180)


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


def writeToTSV(labelFile, pred_results, timePoint = False):
    fileName = labelFile.replace(".tsv", "_events.tsv")
    videoFileName = labelFile.replace(".tsv", ".mp4")
    id = 0
    #className = "shot"
    padding = 1.0 if timePoint else 0 
    dictList = []
    for v in pred_results:
        id += 1
        value = {}

        value['id'] = id
        value['start'] = v[0] - padding if timePoint else  v[0] - padding
        value['end'] = v[0] + padding if timePoint else v[1] + padding
        value['class'] = v[1] if timePoint else v[2]
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


def getShotAPI(videoFileName, predict_file):
    frameRate = getFrameRate(videoFileName)
    pred_results = findShot(predict_file, frameRate)

    writeToTSV(predict_file, pred_results)

    return pred_results


def calculateF1andWriteResults_1():
    dir = "/mnt/gpu02_raid/data/video/CBA/CBA_demo_v2/"
    labelFiles = "labellist.txt"

    labelFileList = read_file_to_list(dir + labelFiles)
    #predict_file = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_chop/TSV/head350_prediction_1551538896210_sc99_01_q1.tsv"
    #predict_file = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_chop/prediction_1551538896210_sc99_01_q1.tsv"

    for predict_file in labelFileList:
        print("----Processing file: ", predict_file)
        pred_results = findShot(dir + predict_file)

        checkResultsOverlap(pred_results)

        true_results = None
        # if predict_file == "prediction_1551538896210_sc99_01_q1.tsv":
        if predict_file == "1551538896210_sc99_01_q1_pd.tsv":
            true_results = [13, 36, 55, 119, 150, 157,
                            186, 328, 350, 386, 444, 469, 526, 586]

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

def getEventLabelsFromText(labelFile):
    if labelFile is None or not os.path.exists(labelFile):
        return False, []
    
    listOfEvents = read_file_to_list(labelFile)
    
    labels = []
    l = int(len(listOfEvents) / 2) 
    for i in range(l):
        labels.append( (float(listOfEvents[i * 2 + 1]), labelConverter(listOfEvents[i * 2])))
            
    return True, labels

def test_getEventLabelsFromText():
    labelFile = '/mnt/gpu02_raid/data/video/CBA/CBA_demo_v3/shotDunkLabels/CBA2.GTevents.txt'
    print(getEventLabelsFromText(labelFile))

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
        pred_results = findShot(predict_file)

        checkResultsOverlap(pred_results)
        
        ret = True
        if textLabels:
            true_results = getEventLabelsFromText(textLabelFolder + odFileName.replace('tsv', 'GTevents.txt'))
        else:
            #ret, true_results = getVideoAndEventLabels(eventLabelJsonFile, videoFileName)
            ret, true_results = getVideoAndEventLabelsCCTV(eventLabelJsonFile, videoFileName)
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
        groundTruthEvents = False
        if groundTruthEvents:
            writeToTSV(predict_file, true_results, True)
        else:
            writeToTSV(predict_file, pred_results)

    print("====F1 report for all the data: ")
    f1Report(overallPred, overallTrue)
    print(allReports)
    print(allCorrectLabels)

dgx02=True

def getValidationResults():    
    dir = "/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/validation/extracted/" if dgx02 else "/raid/data/video/CBA/CBA_5_test_videos/validation/extracted/"
    odFileList = "odFilelist.txt"
    eventLabelJsonFile = '/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/test/extracted/label/Project_all_corrected_manual.aucvl'
    calculateF1andWriteRes(dir, odFileList, eventLabelJsonFile)

def getTestingResults():
    dir = "/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/test/extracted/"  if dgx02 else "/raid/data/video/CBA/CBA_5_test_videos/test/extracted/"
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
    test_getShotStats()
    #getValidationResults()
    #getTestingResults()
    # testGetDegreeOfTwoPoints()
    #test_getEventLabelsFromText()
    #getMiguTestingResults()