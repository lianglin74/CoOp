import sys
from video.getShotMoments import getShotAPI, getFrameRate
from video.generateTSVFromVideo import generateTSVByFrameList, getBaseFileName, saveFramesByFrameList


def test_writeFrames():
    folderName = "/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/validation/testLocalDetect"
    videoFile = "1552493137730_sc99_01_q2.mp4"
    startFrame = 2369
    endFrame = 2371 + 1

    frameList = range(startFrame, endFrame)
    print(frameList)
    saveFramesByFrameList(folderName, videoFile, frameList)

def writeFrames_2():
  frameList=[3944, 3945, 3946, 3950, 3952, 3955, 8210, 8211, 8212, 8213, 8214, 8589, 8594, 8595, 8596, 8598, 9649, 9650, 9651, 9652, 9653, 9654, 9655, 9656, 9657, 9658, 11980, 11984, 11986, 11987, 11989, 11991, 11993, 11994, 11997, 12001, 12002, 12005]
  frameList = [ v -1 for v in frameList]
  
  folderName = "/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/validation/extracted/"
  videoFile = "1551538896210_sc99_01_q1.mp4"
  saveFramesByFrameList(folderName, videoFile, frameList)

if __name__ == '__main__':    
    #test_writeFrames()
    writeFrames_2()
