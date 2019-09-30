import math
import sys
import random
from video.getShotMoments import getShotAPI
from video.getShotMomentsNew import getFrameRate, getEventLabelsFromText, getVideoAndEventLabels, read_file_to_list
from video.generateTSVFromVideo import generateTSVByFrameList, getBaseFileName

# get n random numbers in [lower, upper]
def getNRandomNumber(lower, upper, n):    
    numSet = set()
    if upper - lower + 1 < n:
        print("Not enough number to sample")
        return numSet

    i = 0
    while i < n:
        number = random.randint(lower, upper)
        if number in numSet:
            continue
        else: 
            numSet.add(number)
            i += 1
    
    return numSet

# using prediction file, and for each video, #total frames is given by "totalFrames" (default to 200)
def randomGetFrameListFromVideoAroundEvents(folderName, videoFile, predictionLabelFile, totalFrames = 200):
    fullVideoFileName = folderName + "/"  + videoFile
    fullPredictionLabelFile = folderName + "/" + predictionLabelFile

    pred_results = getShotAPI(fullVideoFileName, fullPredictionLabelFile)
    fps = getFrameRate(fullVideoFileName)
    numEvents =len(pred_results)

    if numEvents == 0:
        print("No events found. Doing nothing")
        return []

    numFramesPerEvent = int( math.ceil( totalFrames / float( numEvents ) ) )

    allFrameSet = set()

    for event in pred_results:
        startFrame = int(event[0] * fps) 
        endFrame = int(event[1] * fps) + 1
        frameSet = getNRandomNumber(startFrame, endFrame, numFramesPerEvent)
        allFrameSet |= frameSet

    frameListSorted = [v for v in allFrameSet]
    frameListSorted.sort()
    return frameListSorted

# Number of events: 
# CBA1: 44; # CBA2: 29;  400/69 = 6. 69*6=
# NBA1: 16; # NBA2: 70; 400/86= 5. 
# Using true events labeling. 
def randomGetFrameListFromVideoAroundEvents_testing(folderName, videoFile, numFramesPerEvent = 5, textLabel = False):
    fullVideoFileName = folderName + "/"  + videoFile

    print("--Processing file: ", fullVideoFileName)

    if textLabel: 
        textLabelFileFolder = '/mnt/gpu02_raid/data/video/CBA/CBA_demo_v3/shotDunkLabels/'
        gtLabelFile = textLabelFileFolder + videoFile.replace('mp4', 'GTevents.txt')
        true_results = getEventLabelsFromText(gtLabelFile)
    else:
        eventLabelJsonFile = '/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/test/extracted/label/Project_all_corrected_manual.aucvl'
        ret, true_results = getVideoAndEventLabels(eventLabelJsonFile, videoFile)

    fps = getFrameRate(fullVideoFileName)    
    numEvents =len(true_results)

    if numEvents == 0:
        print("No events found. Doing nothing")
        return []

    allFrameSet = set()

    padding = 1.0

    for event in true_results:
        roughShotTime = event[0]
        startTime = max(0, roughShotTime - padding)
        endTime = roughShotTime
        startFrame = int(startTime * fps) 
        endFrame = int(endTime * fps) + 1
        frameSet = getNRandomNumber(startFrame, endFrame, numFramesPerEvent)
        allFrameSet |= frameSet

    frameListSorted = [v for v in allFrameSet]
    frameListSorted.sort()
    return frameListSorted

def test_randomGetFrameListFromVideoAroundEvents():
    folderName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_selected_training/"
    videoFile = "5102222619_5004703696_92.mp4"
    predictionLabelFile = "5102222619_5004703696_92.tsv"
    frameList = randomGetFrameListFromVideoAroundEvents(folderName, videoFile, predictionLabelFile)
    print(frameList)

def writeFramesAroundEvents(folderName, videoFileName, predictionLabelFile, totalFrames):
    frameList = randomGetFrameListFromVideoAroundEvents(folderName, videoFileName, predictionLabelFile)
    generateTSVByFrameList(folderName, videoFileName, frameList)

def test_writeFramesAroundEvents():
    folderName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_selected_training/"
    videoFile = "5102222619_5004703696_92.mp4"
    predictionLabelFile = "5102222619_5004703696_92.tsv"

    writeFramesAroundEvents(folderName, videoFile, predictionLabelFile, totalFrames = 200)

# 5 videos from 25 CBA vidoes used for training 
def writeFramesAroundEventsVideoList():
    folderName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_selected_training/"
    videoFileList = [ '1552055654344_sc99_01.mp4', '1552491820703_sc99_01.mp4', '1552498131120_sc99_01.mp4', '1552495094773_sc99_01.mp4', '5102222619_5004703696_92.mp4' ]

    for videoFileName in videoFileList:
        predictionLabelFile =  getBaseFileName(videoFileName) + '.tsv'
        frameList = randomGetFrameListFromVideoAroundEvents(folderName, videoFileName, predictionLabelFile)
        generateTSVByFrameList(folderName, videoFileName, frameList)


########### Used to generate TSV for person detection results
# 4 videos from Migu testing: 2 CBA, 2 NBA. 
def writeFramesAroundEventsVideoList_testing():
    folderName = "/mnt/gpu02_raid/data/video/CBA/CBA_demo_v3/"
    videoFileList = [ 'CBA1.mp4', 'CBA2.mp4', 'NBA1.mp4', 'NBA2.mp4' ]

    for videoFileName in videoFileList:
        frameList = randomGetFrameListFromVideoAroundEvents_testing(folderName, videoFileName, numFramesPerEvent = (6 if 'CBA' in videoFileName else 5), textLabel = True) 
        generateTSVByFrameList(folderName, videoFileName, frameList)

# 20 videos from validation/testing CBA video segments.  
def writeFramesAroundEventsVideoList_validation():
    folderName = "/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/validation/extracted/"
    odFileList = "odFilelist.txt"
    odFileList = read_file_to_list(folderName + odFileList)

    for odFileName in odFileList:
        videoFileName = odFileName.replace(".tsv", ".mp4")        
        frameList = randomGetFrameListFromVideoAroundEvents_testing(folderName, videoFileName) 
        generateTSVByFrameList(folderName, videoFileName, frameList)
    
def writeFramesAroundEventsVideoList_CBAtesting():
    folderName = "/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/test/extracted/"    
    odFileList = "odFilelist.txt"
    odFileList = read_file_to_list(folderName + odFileList)

    for odFileName in odFileList:
        videoFileName = odFileName.replace(".tsv", ".mp4")        
        frameList = randomGetFrameListFromVideoAroundEvents_testing(folderName, videoFileName) 
        generateTSVByFrameList(folderName, videoFileName, frameList)

if __name__ == '__main__':
    if len(sys.argv) == 5:
        folderName = sys.argv[1]
        videoFile = sys.argv[2]
        predictionLabelFile = sys.argv[3]
        totalFrames = sys.argv[4]
        writeFramesAroundEvents(folderName, videoFile, predictionLabelFile, totalFrames)
    else:
        #writeFramesAroundEventsVideoList()
        #writeFramesAroundEventsVideoList_testing()
        writeFramesAroundEventsVideoList_validation()
        writeFramesAroundEventsVideoList_CBAtesting()

