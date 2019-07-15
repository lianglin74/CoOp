import math
import random
import sys
from video.getShotMoments import getShotAPI, getFrameRate
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

def writeFramesAroundEventsVideoList():
    folderName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_selected_training/"
    videoFileList = [ '1552055654344_sc99_01.mp4', '1552491820703_sc99_01.mp4', '1552498131120_sc99_01.mp4', '1552495094773_sc99_01.mp4', '5102222619_5004703696_92.mp4' ]

    for videoFileName in videoFileList:
        predictionLabelFile =  getBaseFileName(videoFileName) + '.tsv'
        frameList = randomGetFrameListFromVideoAroundEvents(folderName, videoFileName, predictionLabelFile)
        generateTSVByFrameList(folderName, videoFileName, frameList)

if __name__ == '__main__':
    if len(sys.argv) == 5:
        folderName = sys.argv[1]
        videoFile = sys.argv[2]
        predictionLabelFile = sys.argv[3]
        totalFrames = sys.argv[4]
        writeFramesAroundEvents(folderName, videoFile, predictionLabelFile, totalFrames)
    else:
        writeFramesAroundEventsVideoList()
