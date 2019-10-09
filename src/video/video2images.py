import cv2
import os
from video.getShotMomentsNew import read_file_to_list

def writeAllFrames(videoFileFullPath, imagePath = None):
    cap = cv2.VideoCapture(videoFileFullPath)
    
    dir = os.path.dirname(videoFileFullPath)
    videoName = os.path.basename(videoFileFullPath)

    if imagePath is None:
        #imagePath = dir + "/frames_" + videoName + "/images/"
        imagePath = dir + "/frames_" + videoName + "/"
    if not os.path.exists(imagePath):
        os.makedirs(imagePath)

    print("--Start extracting frames from: ", videoFileFullPath)
    # get the total frame count
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    
    print("Total frame: ", totalFrames)

    # get fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using new version: ", fps)
    
    frameIndex = 0
    while True: 
        ret, frame = cap.read()
        if not ret:
            print("Stop getting frame from video: ", videoFileFullPath, ", at frameIndex: ", frameIndex)
            break
        else:
            imageId  = videoName.replace('.mp4', '') + '-%02d' % frameIndex
            cv2.imwrite(imagePath + imageId + '.jpg', frame)

        frameIndex += 1

    cap.release()
    
def test_writeAllFrames():
    #videoFile = '/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/validation/extracted/1551538896210_sc99_01_q3.mp4'
    videoFile = '/mnt/gpu02_raid/data/video/CBA/eventDetection/videosForDunkDetection/train/dunk/NBA2-74_0.mp4'
    writeAllFrames(videoFile)

def videos2Frames_byList():
    videoFileList = "videoList.txt"
    #dir = "/mnt/gpu02_raid/data/video/CBA/eventDetection/videosForDunkDetection/train/dunk/"
    dir = "/mnt/gpu02_raid/data/video/CBA/eventDetection/videosForDunkDetection/train/nonDunk/"
    fileList = read_file_to_list(dir + videoFileList)
    fileList = [dir + f for f in fileList]

    for file in fileList:
        print("--processing ", file)
        writeAllFrames(file)

if __name__ == "__main__":
    #test_writeAllFrames()
    videos2Frames_byList()

