from __future__ import division
import cv2
import json
import os
from qd import tsv_io
from qd.tsv_io import tsv_reader
import sys

skipSmallRectParm = 0
skipPersonsParm = 0
filterPersonsParm = 1

robustModeParm = 0

def preReadFrames(cap, numSeconds, startSecond, fps, robustMode = robustModeParm):
    startFrameIndex = int (fps * startSecond)
    if robustMode:
        print("Slow: using robust mode...")
        for i in range(startFrameIndex):
            cap.read()
    else:
        success = cap.set(cv2.CAP_PROP_POS_MSEC, startSecond * 1000.0)        
        if not success:
            print("starting at: ", startFrameIndex, "; expected: ", startSecond * fps)
    
    startFrameIndex = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) 

    cacheFrames = []
    totalFrames = numSeconds * fps
    for i in range(totalFrames):
        ret, frame = cap.read()
        if not ret:
            print("Error in reading frames")
            exit()
        cacheFrames.append(frame)

    return startFrameIndex, cacheFrames

def testSetFrames(cap):
    for i in range(40): 
        success = cap.set(cv2.CAP_PROP_POS_FRAMES, 39 - i)
        if not success:
            print("Failed at: ", i)
            return
    print("no failure")


def getFrameFromVideo(videoFileName, frameIndex):
    cap = cv2.VideoCapture(videoFileName)
    
    # set frame pos
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
    # read frame
    ret, frame = cap.read()
    if not ret:
        print("Error")
        exit()
    cap.release()

    return frame

def showFramesWithLabels(topDir, labelFileName, video_name, startSecond, endSecond = -1, writeImage=0):
    id2Labels = getID2Labels(topDir + "/" + labelFileName)
    print("len(id2Labels)", len(id2Labels))
    
    sepSign = "$"

    labelIndexStartingFromZero = video_name + sepSign + '0' in id2Labels

    if endSecond == -1:
        endSecond = startSecond + 10.0
    
    if writeImage:
        directory = topDir + "/frames_" + video_name + "/" + \
            "start_" + str(startSecond) + "-end_" + str(endSecond) + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)

    cap = cv2.VideoCapture(topDir + "/" + video_name)
    fps = getFPS(cap)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frame: ", totalFrames)

    #
    #testSetFrames(cap)
    
    startFrame = int(startSecond*fps)
    endFrame = min(int(endSecond*fps), totalFrames)
    
    print("StartFrame: ", startFrame)
    print("EndFrame: ", endFrame)

    startFrameIndex, cacheFrames = preReadFrames(cap, 10, startSecond, fps)
    print("Real startFrame: ", startFrameIndex)
    endFrame = startFrameIndex + len(cacheFrames) - 1
    print("Real endFrame: ", endFrame)

    i = startFrameIndex
    while i <= endFrame:
        # read frame
        # set initial frame  #set frame pos
        orgFrame = cacheFrames[i - startFrameIndex]
        orgImageId = video_name + sepSign + str(i)

        imageId = video_name + sepSign + str(i if labelIndexStartingFromZero else i + 1)
        #imageJPG = cv2.imencode('.jpg',frame)[1]

        # get labels
        if imageId not in id2Labels:
            exit()
        else:
            labels = id2Labels[imageId]

        # if (i == 9649):
        #  print(labels)
        #  print(skipRects(labels))

        frame = drawLabel(orgFrame, labels, skipSmallRect = skipSmallRectParm, skipPersons = skipPersonsParm, filterPersons = filterPersonsParm)
        text = "Frame: " + str(i) + "; second: " + str(i / fps)
        frame = cv2.putText(frame, text, (int(0), int(60)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Store this frame to an image
        #my_video_name = video_name.split(".")[0]
        if writeImage:
            cv2.imwrite(directory + 'frame_'+str(i)+'.jpg', frame)

        #cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE);
        height, width = frame.shape[:2]
        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image', width, height)
        cv2.imshow("Image", frame)

        k = cv2.waitKeyEx(0)
        if k == 65361 or k == 2424832:  # left
            i -= 1
        elif k == 65363 or k == 32 or k == 2555904:  # right or space
            i += 1
        elif k == 27 or k == 113:  # q or esc
            exit()
        elif k == 115: #"s" for saving image
            #cv2.imwrite(topDir + imageId + '.jpg', frame)
            cv2.imwrite(topDir + orgImageId + '.jpg', orgFrame)
            
        #else:
        #  print("Key is", k)

    cap.release()


def getID2Labels(labelFileName):
    t1 = tsv_reader(labelFileName)
    id2Labels = {}
    for v in t1:
        id2Labels[v[0]] = json.loads(v[1])

    return id2Labels


def getFPS(cap):
        # get the total frame count
        # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        fps = round(cap.get(cv2.cv.CV_CAP_PROP_FPS))
        print("Frames per second using OLD version", fps)
    else:
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        print("Frames per second using new version", fps)

    return fps

def areaOfRect(rect):
    return (rect[2] - rect[0]) * (rect[3] - rect[1])

def drawLabel(image, labels, skipSmallRect = 0, skipPersons = 1, filterPersons = 1):
    if skipSmallRect:
        labels = skipRects(labels)
    showLabel = True
    confidenceThreshold = 0.1
    playersThreshold = 12

    # matlab RGB to opencv BRG
    colorMap = [(0, 1, 0), (1, 0, 0), (0.75, 0, 0.75), (0, 0, 1), (0, 0.5, 0), (0, 0.75, 0.75), (0.85, 0.325, 0.098), (0.63, 0.078,
                0.1840), (0.929, 0.6940, 0.1250), (0, 0.447, 0.7410), (0.4660, 0.6740, 0.1880), (0.3010, 0.7450, 0.9330), (0.75, 0.75, 0)]
    # from: http://math.loyola.edu/~loberbro/matlab/html/colorsInMatlab.html
    lenColor = len(colorMap)

    #sort objects by area
    labels.sort(key = lambda instance: areaOfRect(instance['rect']), reverse = True)
    #import pdb; pdb.set_trace()    

    i = 0
    countPlayers = 0
    for v in labels:
        #[{"conf": 0.7487, "obj": 0.7487, "class": "basketball", "rect": [1033.5602188110352, 38.45284080505371, 1070.4107284545898, 68.64767646789551]}, {"conf": 0.8385, "obj": 0.8385, "class": "basketball rim", "rect": [408.87592697143555, 251.28884315490723, 486.58501052856445, 312.7845211029053]}, {"conf": 0.9332, "obj": 0.9332, "class": "backboard", "rect": [355.62933349609375, 127.55272674560547, 458.9013671875, 302.18677520751953]}]

        labelName = v["class"]
        rect = v['rect']
        confidenceScore = v['conf']
        pos = (int(rect[2] + 3.0), int(rect[3] - 3.0))
        if (labelName == "backboard"):
            pos = (int(rect[2] + 3.0), int(rect[1] + 3.0))
        elif (labelName == "basketball"):
            pos = (int(rect[2] + 3.0), int(rect[1] + 15.0))
        elif (labelName == "person"):
            if skipPersons or (filterPersons and (confidenceScore < confidenceThreshold or countPlayers >= playersThreshold)):
                continue
            else:
                countPlayers += 1
                pos = (int(rect[2] + 3.0), int( (rect[1] + rect[3])/2.0))

        if 'conf' in v:
            conf = v['conf']
        else:
            conf = 0.0

        # left":1287,"top":228,"width":89,"height":78,
        # draw bounding box
        left = int(round(rect[0]))
        top = int(round(rect[1]))
        topLeft = (left, top)
        lowerRight = (int(rect[2]), int(rect[3]))

        j = i % lenColor
        color = (int(colorMap[j][2]*255),
                 int(colorMap[j][1]*255), int(colorMap[j][0]*255))
        # print(color)
        i += 1
        thickness = 2

        image = cv2.rectangle(image, topLeft, lowerRight, color, thickness)
  
        if showLabel: 
          text = labelName + ", conf: " + str(conf)
        else:
          text = " "
        image = cv2.putText(
            image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image

# remove the small rects which has big overlap within another rect
def skipRects(labels):
    filterLabels = []

    for v1 in labels:
        #[{"conf": 0.7487, "obj": 0.7487, "class": "basketball", "rect": [1033.5602188110352, 38.45284080505371, 1070.4107284545898, 68.64767646789551]}, {"conf": 0.8385, "obj": 0.8385, "class": "basketball rim", "rect": [408.87592697143555, 251.28884315490723, 486.58501052856445, 312.7845211029053]}, {"conf": 0.9332, "obj": 0.9332, "class": "backboard", "rect": [355.62933349609375, 127.55272674560547, 458.9013671875, 302.18677520751953]}]
        skip = any([rectInRect(v1['rect'], v2['rect'])
                    for v2 in labels if v2 != v1 and v2['class'] == v1['class']])
        if not skip:
            filterLabels.append(v1)

    return filterLabels

# check whether rect0 is in rect1
def rectInRect(rect0, rect1, thresh=0.85):
    '''
    x0, y1, x2, y3
    '''
    w = min(rect0[2], rect1[2]) - max(rect0[0], rect1[0])
    if w < 0:
        return 0
    h = min(rect0[3], rect1[3]) - max(rect0[1], rect1[1])
    if h < 0:
        return 0
    i = w * h
    #a1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    a0 = (rect0[2] - rect0[0]) * (rect0[3] - rect0[1])

    if a0 == 0:
        return True
    else:
        return 1. * i / a0 > thresh


def main():
    topDir = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_demo_v3/"
    #labelFileName = "1551538896210_sc99_01_q1_opd.tsv"
    labelFileName = "CBA2.tsv"

    #labelFileName = "1551538896210_sc99_01_q1.tsv"
    #labelFileName = "temp.tsv"
    video_name = "CBA2.mp4"

    startSecond = 13  # 173.151 #75.151
    endSecond = startSecond + 10

    #startSecond = 385.30
    #endSecond = 387
    showFramesWithLabels(topDir, labelFileName,
                          video_name, startSecond, endSecond)


if __name__ == '__main__':
    if len(sys.argv) == 5:
        topDir = sys.argv[1]
        labelFileName = sys.argv[2]
        video_name = sys.argv[3]
        startSecond = float(sys.argv[4]) 
        showFramesWithLabels(topDir, labelFileName, video_name, startSecond)        
    elif len(sys.argv) == 3:
        fullLabelFileName = sys.argv[1]
        startSecond = float(sys.argv[2])
        topDir = os.path.dirname(fullLabelFileName) + "/"
        labelFileName = os.path.basename(fullLabelFileName) 
        video_name = labelFileName.replace('.tsv', '.mp4')
        showFramesWithLabels(topDir, labelFileName, video_name, startSecond)
    else:
        print("Missing arguments. Usage: python .\labelViewerForVideo.py <dir> <labelFileName> <videoFileName> <startTimeInSecond>")
        print('Example usage: python .\labelViewerForVideo.py /mnt/gpu02_raid/data/video/NBA/0001 NBA_0001_1.tsv NBA_0001_1.mkv 630')        
