import os
from video.getShotMomentsNew import buildEventLists

def extraSegmentsForActionRecogniztion(labelFile, results, timePoint = True):
    videoFileName = labelFile.replace(".tsv", ".mp4")

    videoFileBase = os.path.basename(videoFileName)
    videoFileCore, fileSuffix = videoFileBase.split('.')
    videoFilePath = os.path.dirname(videoFileName)

    if "data/video/CBA/CBA_demo_v3" in videoFilePath:
        prefix = "gs://yaoguang-central-storage/shotDunk/"
        videoFileNamePrefix = videoFileBase[:-4]
    elif "data/video/CBA/CBA_5_test_videos/test/extracted" in videoFilePath:
        prefix = "gs://yaoguang-central-storage/shotDunk/test/"
        videoFileNamePrefix = videoFileBase[:-7]
    else:        
        prefix = "gs://yaoguang-central-storage/shotDunk/tmp/"
        videoFileNamePrefix = videoFileBase[:-7]
    
    testList = ['647b025243d74e719b36d24a0c19df37_sc99', # 39 shot, 1 dunk; 
        'CBA1', #33 shots, 11 dunks;
        'NBA1' #16 shots, 0 dunk
        ]
    # vadalition: 10 dunk; test: 8 dunk; 
    
    if videoFileNamePrefix in testList:
        topDir = '/mnt/gpu02_raid/data/video/CBA/eventDetection/videosForDunkDetection/test/'
    else:
        topDir = '/mnt/gpu02_raid/data/video/CBA/eventDetection/videosForDunkDetection/train/'

    print("events: ", results)

    dictList = buildEventLists(results, timePoint)
    
    videoLength = 2.0

    for value in dictList:
        className = 'nonDunk' if value['class'] != 'dunk' else 'dunk'

        outputVideoDir = topDir + className + "/"
        if os.path.exists(outputVideoDir):
            os.mkdir(outputVideoDir)

        startTime = str(value['start']).replace('.', '_')
        outputVideoName = videoFileCore + '-' + startTime + '.' + fileSuffix
        extractSegment(videoFileName, outputVideoDir + outputVideoName, value['start'], videoLength)


def extractSegment(sourceVideo, outputVideoName, startTime, length):
    # The first command does not work. 
    #myCmd = 'ffmpeg -ss ' + str(startTime) + ' -t ' + str(length) + ' -i ' + sourceVideo +  '  -acodec copy -vcodec copy ' + outputVideoName
    # The following one worked: 
    myCmd = 'ffmpeg -ss ' + str(startTime) + ' -t ' + str(length) + ' -i ' + sourceVideo +  ' -vcodec h264 -acodec aac -strict -2  ' + outputVideoName
    print("Calling: ", myCmd)
    os.system(myCmd)

def extractSegmentPy(sourceVideo, outputVideoName, startTime, length):
    # The first command does not work. 
    #myCmd = 'ffmpeg -ss ' + str(startTime) + ' -t ' + str(length) + ' -i ' + sourceVideo +  '  -acodec copy -vcodec copy ' + outputVideoName
    # The following one worked: 

    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    ffmpeg_extract_subclip(sourceVideo, startTime, startTime + length, targetname=outputVideoName)


def test_extractSegment():
    sourceVideo = '/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/test/extracted/5102216385_5004650317_92_q3.mp4'
    outputVideoName = '/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/test/extracted/5102216385_5004650317_92_q3_extracted_test_s2_simple-19.mp4'
    startTime = 19.120
    length = 2

    extractSegment(sourceVideo, outputVideoName, startTime, length)
    #extractSegment(sourceVideo, outputVideoName, startTime, length)

if __name__ == '__main__':
    test_extractSegment()