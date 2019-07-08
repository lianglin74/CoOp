try:
  from itertools import izip as zip
except:
  pass

import json
from qd import tsv_io
from qd.tsv_io import *
import os
import sys

def splitByVideoIds(tsvOrgFile):
  debug = 0

  dir = os.path.dirname(tsvOrgFile)
  tsvFileName = os.path.basename(tsvOrgFile)
  
  trainFileName = dir + "/train-" + tsvFileName
  testFileName =  dir + "/test-" + tsvFileName

  videoToTestFile = ['1551538896210_sc99_01.mp4', '647b025243d74e719b36d24a0c19df37_sc99_.mp4', '5102216385_5004650317_92.mp4', '1552493137730_sc99_01.mp4', '1552141947911_sc99_01.mp4']
  #                      3, 29, 27, 20, 14

  fc = open(trainFileName, 'w');
  fw = open(testFileName, 'w');

  lineCnt = 0; 

  t2 = tsv_reader(tsvOrgFile)

  for v2 in t2:
    lineCnt += 1
    if debug and (lineCnt > 16):
      exit()
    
    videoId  = v2[0].replace('$','/').split('/')[1]
    if videoId not in videoToTestFile:
      a = json.loads(v2[1])
      fc.write(v2[0] + "\t" + json.dumps(a) + "\t" + v2[2]+"\n");
    else:
      a = json.loads(v2[1])
      fw.write(v2[0] + "\t" + json.dumps(a) + "\t" + v2[2]+"\n");
      
  fc.close();
  fw.close();

def workOnCBA_0():  
  splitByVideoIds("/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/split_CBA_video_0/CBA_video_0.withLabel.correct.tsv")

if __name__ == '__main__':
  if len(sys.argv) == 2:
    tsvWithLabelFile = sys.argv[1]
    splitByVideoIds(tsvWithLabelFile)
  else:
    print("Please check arguments")


