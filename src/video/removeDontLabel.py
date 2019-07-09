try:
  from itertools import izip as zip
except:
  pass

import json
from qd import tsv_io
from qd.tsv_io import *
import sys


def removeDontLabel(tsvOrgFile, labelFileName, mergedFileName):
  debug = 0
  
  f2 = open(mergedFileName, 'w');

  skipStr = "SKIP"

  lineCnt = 0; 

  t1 = tsv_reader(tsvOrgFile)
  t2 = tsv_reader(labelFileName)

  for v1, v2 in zip(t1, t2):
    lineCnt += 1
    if debug and (lineCnt > 4):
      exit()
            
    assert(v1[0] == v2[0])
    if (v2[1].find(skipStr) == -1 and v2[1].find("NotDone") == -1):
      a = json.loads(v2[1])
      f2.write(v1[0] + "\t" + json.dumps(a) + "\t" + v1[2]+"\n");

  f2.close();

def generateCBA_0():

  tsvOrgFile = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_0.tsv"
  labelFileName = "/home/yaowe/dev/quickdetection/CBA_video_0.label.tsv"

  mergedFileName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_0.withLabel.tsv"
  
  removeDontLabel(tsvOrgFile, labelFileName, mergedFileName)
 
def generateCBA_1():

  tsvOrgFile = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_1.tsv"
  labelFileName = "/home/yaowe/dev/quickdetection/CBA_video_1.label.tsv"

  mergedFileName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_1.withLabel.tsv"
  
  removeDontLabel(tsvOrgFile, labelFileName, mergedFileName)

def generateCBA_2_part01():

  tsvOrgFile = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_2.tsv"
  labelFileName = "/home/yaowe/dev/quickdetection/CBA_video_2_p01.label.tsv"

  mergedFileName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_2_p01.withLabel.tsv"
  
  removeDontLabel(tsvOrgFile, labelFileName, mergedFileName)
  
if __name__ == '__main__':
  #generateCBA_2_part01()
  if len(sys.argv) == 4:
    tsvOrgFile = sys.argv[1]
    labelFileName = sys.argv[2]    
    mergedFileName = sys.argv[3]
    
    removeDontLabel(tsvOrgFile, labelFileName, mergedFileName)
  else:
    print("Please check arguments")

