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
    
    pred_results_angle = []
    
    imageCnt = 0
    
    upperBallLocation = (-1 , -1)
    #startToTrackBall = F
    eventStart = False
    #eventEnd = False
    angleBallToRim = 270/180.0*math.pi
    
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
            
          if iou > rimBallIouLowerThresh:
            currentTime = imageCnt/frameRate
            if 1:
              print("--Processing image: ", imageCnt, "; second: ", currentTime)
              print("Found both rim and ball with iou: ", iou)
            if len(pred_results) == 0 or currentTime - pred_results[-1] > oneShotTimethresh: 
              pred_results.append(int(currentTime)); 
          
            if (iou > rimBallIouHigherThresh): 
              print("!!Found a shot")
              
          ## use ball rim angle to filter out 
          ballCenter = getCenterOfObject(ballRects)
          if debug:
            print("image: ", imageCnt)
            print("second: ", imageCnt / frameRate)
            print("ballCenter: ", ballCenter)
            
          listOfBallToRimsDistance = [ getDistanceOfTwoPoints(  ballCenter, getCenterOfObject(b)) for b in rimRects]
          
          distanceFromBallToClosetRim = min ( listOfBallToRimsDistance )
          indexOfClosestRim = listOfBallToRimsDistance.index( distanceFromBallToClosetRim )
          rectClosestRim = rimRects[indexOfClosestRim]
          widthRim = getWidthOfObject( rectClosestRim )
          centerOfRim = getCenterOfObject(rectClosestRim)
          
          if debug:
            print("eventStart: ", eventStart)
            print("distanceFromBallToClosetRim: ", distanceFromBallToClosetRim)
            print("distance Thresh:", distanceFromBallToRimToTrack * widthRim)
            print("centerOfRim: ", centerOfRim)
          
          if not eventStart: 
            if distanceFromBallToClosetRim < distanceFromBallToRimToTrack * widthRim and isAbove(ballCenter, centerOfRim):
              angleBallToRim = getAngleOfTwoPoints(ballCenter, centerOfRim)
              eventStart = True
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
              
              if abs(angleRimToBall - angleBallToRim) < angleThresh:                 
                  print("Finding one shot by angle analysis: ", int(imageCnt/frameRate))                
              else: #not a shot
                if debug:
                  print("Warning: possible wrong")
              
            else:
              if debug:
                print("Skipping")
          ## end 
          
    return pred_results
        
def toDegree(angle):
  return angle / math.pi * 180.0; 
  
def calculateF1(pred_results, true_results):
  oneShotTimethresh = 5
  
  lastTime = true_results[-1]
  
  filtered_pred_results = [ v for v in pred_results if v < lastTime + oneShotTimethresh]
  
  print("filtered_pred_results: ", filtered_pred_results, len(filtered_pred_results))
  print("true_results: ", true_results, len(true_results))
  
  truePositiveList = intersection(filtered_pred_results, true_results)
  print("TruePositve: ", truePositiveList)
  #trueNegative = 
  falsePositiveList = [value for value in filtered_pred_results if value not in true_results]
  print("FlasePositve: ", falsePositiveList)
  falseNegativeList = [value for value in true_results if value not in filtered_pred_results]
  print("falseNegative: ", falseNegativeList)
  precision = len(truePositiveList) / (len(truePositiveList) + len (falsePositiveList) +0.0)
  
  recall = len(truePositiveList) / (len(truePositiveList) + len (falseNegativeList) +0.0)
  
  print("precision: ", precision)
  print("recall: ", recall)

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

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
    
def main():
  #predict_file = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_chop/TSV/head350_prediction_1551538896210_sc99_01_q1.tsv"
  predict_file = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_chop/prediction_1551538896210_sc99_01_q1.tsv"
  
  pred_results =  findShot(predict_file)
  true_results = [13, 36, 55, 119, 150, 157, 186, 328, 350, 386]
  
  calculateF1(pred_results, true_results)

if __name__ == '__main__':
  main()
  #testGetDegreeOfTwoPoints()
