import cv2
import os


def writeAllFrames(videoFileFullPath, imagePath = None):
    cap = cv2.VideoCapture(videoFileFullPath)
    
    dir = os.path.dirname(videoFileFullPath)
    videoName = os.path.basename(videoFileFullPath)

    if imagePath is None:
        imagePath = dir + "/frames_" + videoName + "/images/"
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
            imageId  = '%07d' % frameIndex
            cv2.imwrite(imagePath + imageId + '.jpg', frame)

        frameIndex += 1

    cap.release()
    
def test_writeAllFrames():
    #videoFile = '/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/validation/extracted/1551538896210_sc99_01_q3.mp4'
    videoFile = '/mnt/gpu02_raid/data/video/CBA/eventDetection/videosForDunkDetection/train/dunk/NBA2-74_0.mp4'
    writeAllFrames(videoFile)


if __name__ == "__main__":
    test_writeAllFrames()

