import json
from qd import tsv_io
from qd.tsv_io import tsv_reader
from video.labelViewerForVideo import getID2Labels, skipRects, areaOfRect
from video.getShotMomentsNew import read_file_to_list
import sys

def genTSVForPersonDetection(topDir, video_name, tsv_images, tsv_labels, out_tsv):
    skipSmallRects = False
    
    id2Labels = getID2Labels(topDir + "/" + tsv_labels)
    print("len(id2Labels)", len(id2Labels))

    sepSign = "$"    
    labelIndexStartingFromZero = video_name + sepSign + '0' in id2Labels

    t1 = tsv_reader(topDir + "/" + tsv_images)

    f = open(topDir + "/" + out_tsv, 'w')
    
    for v1 in t1:    
        coded_image = v1[2]
        
        imageId = v1[0]

        if not labelIndexStartingFromZero:
            print("The starting index from two TSVs are different! Exiting...")
            exit()
        
        # get labels
        if imageId not in id2Labels:
            print("Cannot find imageID. Exiting...")
            exit()
        else:
            labels = id2Labels[imageId]

        if skipSmallRects:        
          labels= skipRects(labels)

        labels = filterPersonLabel(labels)

        f.write(imageId + '\t' + json.dumps(labels) + '\t' + coded_image + "\n")

    f.close()
        
def filterPersonLabel(labels):    
    filterLabels = []

    confidenceThreshold = 0.7
    playersThreshold = 12

    # matlab RGB to opencv BRG
    colorMap = [(0, 1, 0), (1, 0, 0), (0.75, 0, 0.75), (0, 0, 1), (0, 0.5, 0), (0, 0.75, 0.75), (0.85, 0.325, 0.098), (0.63, 0.078,
                0.1840), (0.929, 0.6940, 0.1250), (0, 0.447, 0.7410), (0.4660, 0.6740, 0.1880), (0.3010, 0.7450, 0.9330), (0.75, 0.75, 0)]
    # from: http://math.loyola.edu/~loberbro/matlab/html/colorsInMatlab.html
    lenColor = len(colorMap)

    #sort objects by area
    labels.sort(key = lambda instance: areaOfRect(instance['rect']), reverse = True)
    
    countPlayers = 0
    for v in labels:
        #[{"conf": 0.7487, "obj": 0.7487, "class": "basketball", "rect": [1033.5602188110352, 38.45284080505371, 1070.4107284545898, 68.64767646789551]}, {"conf": 0.8385, "obj": 0.8385, "class": "basketball rim", "rect": [408.87592697143555, 251.28884315490723, 486.58501052856445, 312.7845211029053]}, {"conf": 0.9332, "obj": 0.9332, "class": "backboard", "rect": [355.62933349609375, 127.55272674560547, 458.9013671875, 302.18677520751953]}]
        labelName = v["class"]
        rect = v['rect']
        confidenceScore = v['conf']
        
        if (labelName == "person") and confidenceScore >= confidenceThreshold:            
            if countPlayers >= playersThreshold:
                break

            j = countPlayers % lenColor
            color = [int(colorMap[j][2]*255), int(colorMap[j][1]*255), int(colorMap[j][0]*255)]
            v['color'] = color
            filterLabels.append(v)
            countPlayers += 1

    return filterLabels

def genTSV_forMiguTesting():
    topDir = "/mnt/gpu02_raid/data/video/CBA/CBA_demo_v3/"
    videoFileList = [ 'CBA1.mp4', 'CBA2.mp4', 'NBA1.mp4', 'NBA2.mp4' ]

    for video_name in videoFileList:
        #video_name = "CBA1.mp4"
        videoFileBase = video_name.replace(".mp4", "")
        #tsv_images = "extracted_from_CBA1.tsv"
        tsv_images = "extracted_from_" + videoFileBase + ".tsv"
        tsv_labels = videoFileBase + "_people.tsv"

        out_tsv = videoFileBase + "_player_for_vendor.tsv"

        genTSVForPersonDetection(topDir, video_name, tsv_images, tsv_labels, out_tsv)

def genTSV_forCBA_validation():
    topDir = "/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/validation/extracted/extractPersonDetectionFrames/"
    odFileList = "odFilelist.txt"
    odFileList = read_file_to_list(topDir + odFileList)
    videoFileList = [ v.replace(".tsv", ".mp4") for v in odFileList]

    for video_name in videoFileList:
        #video_name = "CBA1.mp4"
        videoFileBase = video_name.replace(".mp4", "")
        #tsv_images = "extracted_from_CBA1.tsv"
        tsv_images = "extracted_from_" + videoFileBase + ".tsv"
        tsv_labels = videoFileBase + "_people.tsv"

        out_tsv = videoFileBase + "_player_for_vendor.tsv"

        genTSVForPersonDetection(topDir, video_name, tsv_images, tsv_labels, out_tsv)

def genTSV_forCBA_testing():
    topDir = "/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/test/extracted/extractPersonDetectionFrames/"
    odFileList = "odFilelist.txt"
    odFileList = read_file_to_list(topDir + odFileList)
    videoFileList = [ v.replace(".tsv", ".mp4") for v in odFileList]

    for video_name in videoFileList:
        #video_name = "CBA1.mp4"
        videoFileBase = video_name.replace(".mp4", "")
        #tsv_images = "extracted_from_CBA1.tsv"
        tsv_images = "extracted_from_" + videoFileBase + ".tsv"
        tsv_labels = videoFileBase + "_people.tsv"

        out_tsv = videoFileBase + "_player_for_vendor.tsv"

        genTSVForPersonDetection(topDir, video_name, tsv_images, tsv_labels, out_tsv)


if __name__ == '__main__':
    #genTSV_forCBA_validation()
    genTSV_forCBA_testing()