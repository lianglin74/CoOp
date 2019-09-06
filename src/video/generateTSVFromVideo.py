import numpy as np
import cv2
import time
import base64
import json
import random


def getVideoList(videoListFileName):
    with open(videoListFileName) as f:
        videoList = f.read().splitlines()
    return videoList


def generateTSV(folderName, videoList, totalFramePerVideo, totalBatch, allFrames=False):
    tsvFolderName = "TSV"
    # totalFramePerVideo = 4;
    # totalBatch = 2;

    debug = 1

    sepSign = "$"
    idSet = set()
    if not allFrames:
        numImageInBatch = totalFramePerVideo/totalBatch

    emptyJson = json.dumps([])

    startTime = time.process_time()
    print("startTime: ", startTime)

    for batch in range(totalBatch):
        tsvFileName = tsvFolderName + "/" + \
            folderName + '_video_' + str(batch) + '.tsv'
        # open the tsv file
        f = open(tsvFileName, 'w')

        videoCnt = 0
        for video_name in videoList:
            videoCnt += 1
            if debug:
                print("--Batch: ", batch,  " ; Processing ",
                      videoCnt, "-th video: ", video_name)
            cap = cv2.VideoCapture(folderName + '/' + video_name)

            # get the total frame count
            totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if debug:
                print("Total frame: ", totalFrames)

            if allFrames:
                numImageInBatch = totalFrames / totalBatch + 1
                frameIndex = batch

            i = 0
            while i < numImageInBatch and frameIndex < totalFrames:
                if debug and i % 100 == 0:
                    print("Processing ", i, "-th frame")
                    endTime = time.process_time()
                    print("Used time: ", endTime - startTime)
                # for imageIndex in range(numImageInBatch):
                # randomly generate the frames
                if not allFrames:
                    frameIndex = random.randint(0, totalFrames-1)
                # generate id
                imageId = folderName + '/' + \
                    video_name + sepSign + str(frameIndex)
                if imageId not in idSet:
                    idSet.add(imageId)
                    i += 1
                else:
                    continue

                # set frame pos
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
                # read frame
                ret, frame = cap.read()
                if not ret:
                    print("Error")
                    exit()

                if allFrames:
                    frameIndex += totalBatch

                imageJPG = cv2.imencode('.jpg', frame)[1]

                # write to tsv
                f.write(imageId + '\t' + emptyJson + '\t')
                img64 = (base64.b64encode(imageJPG)).decode('ascii')
                f.write(img64)
                f.write('\n')

                # Store this frame to an image
                # my_video_name = video_name.split(".")[0]
                # cv2.imwrite(my_video_name+'_frame_'+str(frameIndex)+'.jpg',frame)

            cap.release()

        f.close()

    endTime = time.process_time()
    print("endTime: ", endTime)
    print("Total used time: ", endTime - startTime)


def getBaseFileName(fileName):
    index = fileName.rfind('.')
    return fileName if index == -1 or index == 0 else fileName[0: index]

def read_file_to_list(file_name):
    res_lists = []
    with open(file_name, 'r') as file:  # Use file to refer to the file object
        data = file.read()
        res_lists = data.split()

    return res_lists

def generateTSVByFrameList(folderName, videoFileName, frameList):
    debug = 0
    sepSign = "$"
    emptyJson = json.dumps([])

    startTime = time.process_time()
    print("startTime: ", startTime)

    tsvFileName = folderName + "/" + 'extracted_from_' + \
        getBaseFileName(videoFileName) + '.tsv'
    # open the tsv file
    f = open(tsvFileName, 'w')

    cap = cv2.VideoCapture(folderName + '/' + videoFileName)
    
    for frameIndex in frameList:
        imageId = videoFileName + sepSign + str(frameIndex)
    
        # set frame pos
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        # read frame
        ret, frame = cap.read()
        if not ret:
            print("Error")
            exit()

        imageJPG = cv2.imencode('.jpg', frame)[1]

        # write to tsv
        f.write(imageId + '\t' + emptyJson + '\t')
        img64 = (base64.b64encode(imageJPG)).decode('ascii')
        f.write(img64)
        f.write('\n')

    cap.release()
    f.close()

    endTime = time.process_time()
    print("endTime: ", endTime)
    print("Total used time: ", endTime - startTime)

def saveFramesByFrameList(folderName, videoFileName, frameList, returnFrames = 0):
    sepSign = '$'
    cap = cv2.VideoCapture(folderName + '/' + videoFileName)
    imageList = []
    
    for frameIndex in frameList:
        imageId = videoFileName + sepSign + str(frameIndex)
    
        # set frame pos
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        # read frame
        ret, frame = cap.read()
        if not ret:
            print("Error")
            exit()

        if returnFrames:
            imageList.append(frame)
        else:
            cv2.imwrite(imageId +'.jpg', frame)

    cap.release()
    if returnFrames:
        return imageList

def genTSVForCBA():
    folderName = "CBA"
    videoListFileName = folderName + '/' + 'CBAlist.txt'
    generateTSV(folderName, getVideoList(videoListFileName), 2500, 10)


def genTSVForDunk():
    folderName = "Dunk"
    videoListFileName = folderName + '/' + 'Dunklist.txt'
    generateTSV(folderName, getVideoList(videoListFileName), 0, 3, True)


if __name__ == '__main__':
    # genTSVForCBA();
    genTSVForDunk()
