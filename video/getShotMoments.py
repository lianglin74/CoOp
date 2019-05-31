try:
  from itertools import izip as zip
except:
  pass
import json
from qd import tsv_io
from qd.process_tsv import onebb_in_list_of_bb
from qd.tsv_io import tsv_reader, tsv_writer
from qd.qd_common import calculate_iou

from tqdm import tqdm
import numpy as np
import math

def findShot(predict_file):
    
    basketBallThresh = 0.5
    rimThresh = 0.2
    backBoardThresh = 0.1
    rimBallIouLowerThresh = 0.001
    rimBallIouHigherThresh = 0.20
    
    # the time period between two shots
    oneShotTimethresh = 1
    
    # 2.0 * rim width
    distanceFromBallToRimToTrack = 2.0
    angleThresh = 100.0/180*math.pi
	
    debug = 0
    
    bufferSeconds = 2    
    frameRate = 25.0
    totalBufferingFrames = 50
    #buffer for 3 seconds? 
    pred_key_to_rects = {}
    
    pred_results = []
    iou_list = []
    ball_above_rim = []
    
    pred_results_angle = []
    
    imageCnt = 0
    
    upperBallLocation = (-1 , -1)
    #startToTrackBall = F
    eventStart = False
    #eventEnd = False
    angleBallToRim = 270/180.0*math.pi
    padding = 1.0
    
    startTime  = -1
    endTime = -1
    
    iouTime = -1
    
    for row in tqdm(tsv_reader(predict_file)):
        imageCnt += 1
        
        if (imageCnt > 875 and imageCnt < 950):
          debug = 0
        
        key = row[0]
        rects = json.loads(row[1])
        
        rimExists = False
        ballExists = False
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
              #filteredRects.append(r)
              if debug and not rimExists:
                print("[Warning] image ",  imageCnt, ": backboard found, but no rim")              
          i += 1
        if maxBasketRectIndex != -1:
          ballRects = rects[maxBasketRectIndex]          
        
        if debug and imageCnt == 344:
          print("Rects: " , rects)
          print("rimExists: " , rimExists)
          print("ballExists: " , ballExists)
        
        if rimExists and ballExists:          
          assert ballRects['class'] == "basketball"          
            
          iou = maxIouOnebb_in_list_of_bb(ballRects, rimRects)
          
          if debug and imageCnt == 400:
            print("ballRects:", ballRects)
            print("rimRect:", rimRects)
            print("iou:", iou)
            #
            exit()          
          
          ## use ball rim angle to filter out 
          ballCenter = getCenterOfObject(ballRects)
            
          listOfBallToRimsDistance = [ getDistanceOfTwoPoints(  ballCenter, getCenterOfObject(b)) for b in rimRects]
          
          distanceFromBallToClosetRim = min ( listOfBallToRimsDistance )
          indexOfClosestRim = listOfBallToRimsDistance.index( distanceFromBallToClosetRim )
          rectClosestRim = rimRects[indexOfClosestRim]
          widthRim = getWidthOfObject( rectClosestRim )
          centerOfRim = getCenterOfObject(rectClosestRim)
          
          if iou > rimBallIouLowerThresh:
            currentTime = imageCnt/frameRate
            iouTime = currentTime
            if 1:
              print("--Processing image: ", imageCnt, "; second: ", currentTime)
              print("Found both rim and ball with iou: ", iou)
              print("Ball is above rim: ", isAbove(ballCenter, centerOfRim))
            if len(pred_results) == 0 or currentTime - pred_results[-1][1] > oneShotTimethresh: 
              pred_results.append((currentTime, currentTime + oneShotTimethresh));
              iou_list.append(iou)
              ball_above_rim.append(isAbove(ballCenter, centerOfRim))
          
            if (iou > rimBallIouHigherThresh): 
              print("!!Found a shot")
              
          
          if debug:
            print("image: ", imageCnt)
            print("second: ", imageCnt / frameRate)
            print("ballCenter: ", ballCenter)
            
          if debug:
            print("eventStart: ", eventStart)
            print("distanceFromBallToClosetRim: ", distanceFromBallToClosetRim)
            print("distance Thresh:", distanceFromBallToRimToTrack * widthRim)
            print("centerOfRim: ", centerOfRim)
          
          
          if not eventStart: 
            if distanceFromBallToClosetRim < distanceFromBallToRimToTrack * widthRim and isAbove(ballCenter, centerOfRim):
              angleBallToRim = getAngleOfTwoPoints(ballCenter, centerOfRim)
              eventStart = True
              startTime = imageCnt/frameRate
              if debug:
                print("---key event start")
                print("angleBallToRim: ", toDegree(angleBallToRim))
          else: #event started
            if distanceFromBallToClosetRim >= distanceFromBallToRimToTrack * widthRim:
              eventStart = False
              if debug:
                print("~~key event end")
            elif not isAbove(ballCenter, centerOfRim):
              eventStart = False; 
              
              angleRimToBall = getAngleOfTwoPoints(centerOfRim, ballCenter)
              
              if debug:
                print("~~event ends: ")
                print("realAngle: ", toDegree(angleRimToBall))
                print("relative angle: ", toDegree(abs(angleRimToBall - angleBallToRim)))
              
              endTime = imageCnt/frameRate
              if abs(angleRimToBall - angleBallToRim) < angleThresh and iouTime > startTime - padding and iouTime < endTime + padding:     
                  
                  print("Finding one shot by angle analysis: ", (imageCnt/frameRate))
                  pred_results_angle.append((startTime - padding, endTime + padding))
              else: #not a shot
                if debug:
                  print("Warning: possible wrong")
              
            else:
              if debug:
                print("Skipping")
          ## end 
          
    print(pred_results)
    print(iou_list)
    print(ball_above_rim)
    print(pred_results_angle)
    
    return pred_results_angle
    #pred_results
        
def toDegree(angle):
  return angle / math.pi * 180.0; 
  
def calculateF1(pred_results, true_results):
  oneShotTimethresh = 5
  
  lastTime = true_results[-1]
  
  filtered_pred_results = pred_results
  #[ v for v in pred_results if v < lastTime + oneShotTimethresh]
  
  print("filtered_pred_results: ", filtered_pred_results, len(filtered_pred_results))
  print("true_results: ", true_results, len(true_results))
  
  truePositiveList = [value for value in filtered_pred_results if findPairInValueList(value, true_results)]
  print("TruePositve: ", truePositiveList)
  #trueNegative = 
  falsePositiveList = [value for value in filtered_pred_results if not findPairInValueList(value, true_results)]
  print("FlasePositve: ", falsePositiveList)
  falseNegativeList = [value for value in true_results if not findValueInPairList( value ,  filtered_pred_results) ]
  print("falseNegative: ", falseNegativeList)
  precision = len(truePositiveList) / (len(truePositiveList) + len (falsePositiveList) +0.0)
  
  recall = len(truePositiveList) / (len(truePositiveList) + len (falseNegativeList) +0.0)
  
  print("precision: ", precision)
  print("recall: ", recall)

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

def findPairInValueList(pair, true_results):
  return any([v >= pair[0] and v <= pair[1]  for v in true_results])

def findValueInPairList(v, filtered_pred_results):
  return any([v >= pair[0] and v <= pair[1]  for pair in filtered_pred_results])

def isAbove(p1, p2):  
  return p1[1] < p2[1]
        
        
def getWidthOfObject(bb):
  y1 = bb['rect'][1]
  
  y2 = bb['rect'][3]
  
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

#from p1 pointing to p2. Good to use
def getAngleOfTwoPoints(p1, p2):
    deltaX =  -(p1[0] - p2[0])
    len = np.linalg.norm(np.array(p1) - np.array(p2))
    orgAngle = np.arccos(deltaX/ len)
    if (p1[1] > p2[1]): 
      orgAngle = math.pi * 2.0 - orgAngle
    return orgAngle
        
def maxIouOnebb_in_list_of_bb(bb, bbs):
    return max([calculate_iou(b['rect'], bb['rect']) for b in bbs])

def testGetDegreeOfTwoPoints_v1():
  listOfPoints =[[0, 0], [0, 1], [1, 2], [2, 2], [2, 1], [1, 0]]
  for v in listOfPoints:
    p1 = np.array(v)
    p2 = np.array((1, 1))
    print("p1 is: ", p1)
    print("Angle is: ", angle_between(p1, p2) / math.pi * 180)

def testGetDegreeOfTwoPoints():
  listOfPoints =[[0, 0], [0, 1], [1, 2], [2, 2], [2, 1], [1, 0], [2, 0], [0, 2]]
  for v in listOfPoints:
    p1 = v

    #rim
    p2 = (1, 1)
    print("p1 is: ", p1)
    print("Angle is: ", getAngleOfTwoPoints(p1, p2) / math.pi * 180)

def writeToTSV(labelFile, pred_results):
  fileName = labelFile.replace(".tsv", "_events.tsv")
  videoFileName = labelFile.replace(".tsv", ".mp4")
  id = 0; 
  className = "shot"
  padding  = 0
  dictList = []
  for v in pred_results:
    id += 1
    value = {}
    value['class'] = className
    value['id'] = id
    value['start'] = v[0] - padding
    value['end'] = v[1] + padding    
    dictList.append(value)
  
  print('json format results:')
  print(dictList)
  
  f = open(fileName, 'w')  
  f.write(videoFileName + '\t' + json.dumps(dictList) + '\n');
  f.close()
  
def read_file_to_list(file_name):
  res_lists = []
  with open(file_name, 'r') as file: # Use file to refer to the file object
    data = file.read() 
    res_lists = data.split()    
  
  return res_lists;
  
def main():
  dir = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_chop/"
  labelFiles = "labellist.txt"
  labelFileList = read_file_to_list(dir + labelFiles)
  #predict_file = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_chop/TSV/head350_prediction_1551538896210_sc99_01_q1.tsv"
  #predict_file = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_chop/prediction_1551538896210_sc99_01_q1.tsv"
  for predict_file in labelFileList:
    pred_results =  findShot(dir + predict_file)
    #if predict_file == "prediction_1551538896210_sc99_01_q1.tsv": 
    if predict_file == "1551538896210_sc99_01_q1.tsv": 
      true_results = [13, 36, 55, 119, 150, 157, 186, 328, 350, 386, 444, 469, 526, 586]
      calculateF1(pred_results, true_results)
    
    writeToTSV(predict_file, pred_results)

if __name__ == '__main__':
  main()
  #testGetDegreeOfTwoPoints()
