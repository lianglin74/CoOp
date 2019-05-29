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

def findShot(predict_file):
    
    basketBallThresh = 0.5
    rimThresh = 0.2
    backBoardThresh = 0.1
    rimBallIouLowerThresh = 0.001
    rimBallIouHigherThresh = 0.20
    
    # the time period between two shots
    oneShotTimethresh = 1
    
    debug = 0
    
    bufferSeconds = 2    
    frameRate = 25.0
    totalBufferingFrames = 50
    #buffer for 3 seconds? 
    pred_key_to_rects = {}
    
    pred_results = []
    
    imageCnt = 0
    for row in tqdm(tsv_reader(predict_file)):
        imageCnt += 1
        
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
          
          if debug and imageCnt == 344:
            print("ballRects:", ballRects)
            print("rimRect:", rimRects)
            print("iou:", iou)
            
          if iou > rimBallIouLowerThresh:
            currentTime = imageCnt/frameRate
            print("--Processing image: ", imageCnt, "; second: ", currentTime)
            print("Found both rim and ball with iou: ", iou)
            if len(pred_results) == 0 or currentTime - pred_results[-1] > oneShotTimethresh: 
              pred_results.append(int(currentTime)); 
          
          if (iou > rimBallIouHigherThresh): 
            print("!!Found a shot")
        
    return pred_results
        
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

def maxIouOnebb_in_list_of_bb(bb, bbs):
    return max(calculate_iou(b['rect'], bb['rect']) for b in bbs)

def main():
  #predict_file = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_chop/TSV/head350_prediction_1551538896210_sc99_01_q1.tsv"
  predict_file = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_chop/prediction_1551538896210_sc99_01_q1.tsv"
  
  pred_results =  findShot(predict_file)
  true_results = [13, 36, 55, 119, 150, 157, 186, 328, 350, 386]
  
  calculateF1(pred_results, true_results)

if __name__ == '__main__':
  main()
