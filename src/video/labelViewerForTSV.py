from __future__ import division
import cv2
import json
import os
from qd import tsv_io
from qd.tsv_io import tsv_reader
import sys
import numpy as np
import base64
from video.labelViewerForVideo import getID2Labels, drawLabel


def showImagesWithLabels(topDir, labelFileName, tsvFileName):
    skipSmallRects = False
    
    id2Labels = getID2Labels(topDir + "/" + labelFileName)
    print("len(id2Labels)", len(id2Labels))

    t1 = tsv_reader(topDir + "/" + tsvFileName)
    
    allImages = []
    for v1 in t1:    
      allImages.append(v1)
      
    l = len(allImages)
    i = 0
    while i <= l and i >= 0:
        v1 = allImages[i]
        
        coded_image = v1[2]
        
        img = base64.b64decode(coded_image)
        npimg = np.fromstring(img, dtype=np.uint8)
        frame = cv2.imdecode(npimg, 1)
        
        imageId = v1[0]
        
        # get labels
        if imageId not in id2Labels:
            exit()
        else:
            labels = id2Labels[imageId]

        # if (i == 9649):
        #  print(labels)
        #  print(skipRects(labels))
        if skipSmallRects:        
          frame = drawLabel(frame, skipRects(labels), skipPersons = 0)
        else:
          frame = drawLabel(frame, labels)
        text = "Line: " + str(i + 1) + "; id: " + v1[0]
        frame = cv2.putText(frame, text, (int(0), int(60)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("withBox", frame)

        k = cv2.waitKeyEx(0)
        if k == 65361 or k == 2424832:  # left
            i -= 1
        elif k == 65363 or k == 32 or k == 2555904:  # right or space
            i += 1
        elif k == 27 or k == 113:  # q or esc
            exit()
        #else:
        #  print("Key is", k)


def main():
    #topDir = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_demo_v3/"
    topDir = "/mnt/gpu02_raid/data/video/CBA/CBA_demo_v3/"
    #labelFileName = "1551538896210_sc99_01_q1_opd.tsv"
    tsvFileName = "extracted_from_CBA1.tsv"

    #labelFileName = "1551538896210_sc99_01_q1.tsv"
    #labelFileName = "temp.tsv"
    labelFileName = "CBA1_people.tsv"

    
    showImagesWithLabels(topDir, labelFileName,
                          tsvFileName)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        topDir = sys.argv[1]
        labelFileName = sys.argv[2]
        tsvFileName = sys.argv[3]
        
        showImagesWithLabels(topDir, labelFileName, tsvFileName)
    else:
        main()
        print("Missing arguments. Usage: python .\labelViewerForVideo.py <dir> <labelFileName> <videoFileName> <startTimeInSecond>")
        print('Example usage: python .\labelViewerForVideo.py /mnt/gpu02_raid/data/video/NBA/0001 NBA_0001_1.tsv NBA_0001_1.mkv 630')        
