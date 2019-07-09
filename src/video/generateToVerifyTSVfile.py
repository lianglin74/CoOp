# Purpose: 
# Assume now by comparing prediction results from a trained model and the labeling results from vendor, we got a TSV file with the labels that still need manual verification (srcToVerifyLabelFile). Then we want to split the original TSV file with images and labels to two files: toVerifyTSVFileName and noVerifyTSVFileName. so that a person can manually verifiy the labeling results in the toVerifyTSVFileName. 
import json
from qd import tsv_io
from qd.tsv_io import *
import sys

def generateToVerifyTSVfile(tsvWithLabelFile, srcToVerifyLabelFile, toVerifyTSVFileName, noVerifyTSVFileName):
  debug = 0
  
  lineCnt = 0; 

  t1 = tsv_reader(tsvWithLabelFile)
  
  t2 = tsv_reader(srcToVerifyLabelFile)  
  toVerifyLineList = [ int(v2[0]) for v2 in t2 ]
  
  if debug:
    print toVerifyLineList[0:5]
  
  f2 = open(toVerifyTSVFileName, 'w');
  f1 = open(noVerifyTSVFileName, 'w');

  for v1 in t1:
    lineCnt += 1
    
    if debug and (lineCnt > 11):
      exit()
    
    if lineCnt in toVerifyLineList:
      a = json.loads(v1[1])
      f2.write(v1[0] + "\t" + json.dumps(a) +  "\t" + v1[2] + "\n");
    else:
      a = json.loads(v1[1])
      f1.write(v1[0] + "\t" + json.dumps(a) +  "\t" + v1[2] + "\n");

  f1.close()
  f2.close();
  
def generateToVerifyCBA_2_part23():
  tsvWithLabelFile = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_2_part23/CBA_video_2_p23.withLabel.tsv"
  
  srcToVerifyLabelFile = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_2_part23/srcToVerify_CBA_video_2_p23.pureLabel.tsv"
  toVerifyTSVFileName =  "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_2_part23/toVerify_CBA_video_2_p23.withLabel.tsv"
  noVerifyTSVFileName=  "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_2_part23/noVerify_CBA_video_2_p23.withLabel.tsv"
  generateToVerifyTSVfile(tsvWithLabelFile, srcToVerifyLabelFile, toVerifyTSVFileName, noVerifyTSVFileName)

if __name__ == '__main__':
  generateToVerifyCBA_2_part23()
