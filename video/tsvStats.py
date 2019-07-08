import json
from qd import tsv_io
from qd.tsv_io import *
import sys

def tsvStats(labelFileName, statFileName = None):
  debug = 0

  lineCnt = 0; 

  t1 = tsv_reader(labelFileName)
  
  total = 0
  labelCount = {'empty': 0, 'basketball': 0, 'rim': 0, 'backboard': 0,  'ball+rim': 0, 'all': 0}
  
  if statFileName is not None:
    f = open(statFileName, 'w');
  
  for v1 in t1:
    total += 1
    
    lineCnt += 1
    if debug and (lineCnt > 11):
      break
    
    a = json.loads(v1[1])
    if (len(a) == 0):
      labelCount['empty'] +=1
    
    #ball, rim, backboard
    countAll = [0, 0, 0]
    
    for l in a:
      #print(l['class'])
      #import pdb      
      #pdb.set_trace()
      
      if l['class'] == 'basketball':
        labelCount['basketball'] +=1
        countAll[0] = 1
      elif l['class'] == 'basketball rim':
        labelCount['rim'] +=1
        countAll[1] = 1
      elif l['class'] == 'backboard':
        labelCount['backboard'] +=1
        countAll[2] = 1
      else:
        print('error');
      
      if (countAll[0] and countAll[1]):        
        labelCount['ball+rim'] +=1
      if (countAll[0] and countAll[1] and countAll[2]):
        labelCount['all'] +=1
        
  #import pdb      
  #pdb.set_trace()
  if statFileName is None:
    print("Total is: " + str(total) + '\n')
  else:
    f.write("Total is: " + str(total) + '\n')
  
  for k, v in labelCount.items():
    if statFileName is None:
      print("Count for class " + k + " is: " + str(v) + ", percentage:" + str(v*100.0/total)+ '\n')
    else:
      f.write("Count for class " + k + " is: " + str(v) + ", percentage:" + str(v*100.0/total)+ '\n')
  
  if statFileName is not None:
    f.close()


def CBA_2_part01_correct():
  labelFileName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_2_part01/CBA_video_2_p01.pureLabel.correct.tsv"
  statFileName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_2_part01/CBA_video_2_p01.pureLabel.correct.stats.txt"
  tsvStats(labelFileName, statFileName)

def CBA_correct_0():  
  labelFileName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/split_CBA_video_0/CBA_video_0.pureLabel.correct.tsv"
  statFileName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/split_CBA_video_0/CBA_video_0.pureLabel.correct.stats.txt"
  tsvStats(labelFileName, statFileName)


def CBA_correct_1():
  labelFileName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/split_CBA_video_1/CBA_video_1.pureLabel.correct.tsv"
  statFileName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/split_CBA_video_1/CBA_video_1.pureLabel.correct.stats.txt"
  tsvStats(labelFileName, statFileName)

if __name__ == '__main__':
  if len(sys.argv) == 2:
    tsvStats(sys.argv[1])
  elif len(sys.argv) == 3:
    tsvStats(sys.argv[1], sys.argv[2])
  else:
    print("Please check arguments")
