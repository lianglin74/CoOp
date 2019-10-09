import json
import os

def getVideoAndEventLabels(jsonFile, videoFileName):
    ret = False
    labelList = []

    if jsonFile is None or not os.path.exists(jsonFile):
        return ret, labelList
    with open(jsonFile) as json_file:
        data = json.load(json_file)
        if "data" in data and videoFileName in data['data']:            
            timeLabelList = data['data'][videoFileName]
            ret = True
            labelList = [ (v['time'], labelConverter(v['label'])) for v in timeLabelList]
    
    labelList.reverse()

    return ret, labelList

def labelConverter(label):
    if "dunk" in label:
        return "dunk"
    else:
        return "shot"

def testGetVideoAndEventLabels():
    jsonFile = '/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/test/extracted/label/Project_all.aucvl'
    videoFileName = '1551538896210_sc99_01_q1.mp4'
    #path: /mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/validation/extracted/

    ret, labelList = getVideoAndEventLabels(jsonFile, videoFileName)

    print(labelList)
    #[(13.922896, 'shot'), (36.051405, 'shot'), (55.823736, 'shot'), (120.206053, 'shot'), (151.343382, 'shot')...

def writeNewJsonFileWithCorrectedTime(oldJsonFile, newJsonFile, correctedLabels):
    with open(oldJsonFile) as json_file:
        data = json.load(json_file)
        for video, shotTime in correctedLabels.items():
            #shotTime: {orgTime1: newTime1, orgTime2: newTime2, ...} 
            data['data'][video] = [ {"time": shotTime[v["time"]] if v["time"] in shotTime else v["time"], "label": v["label"]} for v in data['data'][video]]
    
    with open(newJsonFile, 'w') as outfile: 
        json.dump(data, outfile)

def test_writeNewJsonFileWithCorrectedTime():
    oldJsonFile = '/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/test/extracted/label/Project_all.aucvl'
    
    newJsonFile = '/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/test/extracted/label/Project_all_corrected.aucvl'
    
    # for validation videos
    correctedLabels = {'1551538896210_sc99_01_q1.mp4': {13.922896: 13.76, 36.051405: 36.12, 55.823736: 55.24, 120.206053: 120.24, 151.343382: 151.2, 158.211654: 158.0, 186.743713: 186.8, 328.851485: 328.72, 344.177011: 343.96, 350.888363: 350.84, 386.284587: 386.04, 443.872138: 444.12, 469.697181: 469.68, 479.717574: 479.64, 526.101491: 526.16, 586.654381: 586.4}, '1551538896210_sc99_01_q2.mp4': {31.747275: 31.0, 187.875039: 187.08, 363.137339: 362.28, 377.383004: 376.44, 390.52672: 389.6}, '1551538896210_sc99_01_q3.mp4': {12.395167: 12.48, 40.145257: 41.08, 62.80545: 63.08, 80.463042: 80.6, 119.650773: 119.68, 133.792995: 133.76, 159.649808: 159.88, 184.752173: 184.96}, '1551538896210_sc99_01_q4.mp4': {22.023613: 21.08, 35.699745: 34.68, 67.569266: 67.4, 122.508828: 122.32, 157.646352: 157.8, 168.876039: 169.08, 211.513257: 211.04, 335.170871: 335.48, 353.0663: 353.32}, '1552493137730_sc99_01_q1.mp4': {81.088538: 81.08, 95.213236: 95.28, 109.453214: 109.4, 134.672461: 134.84, 147.532706: 147.72, 281.708898: 281.56, 297.787567: 297.84}, '1552493137730_sc99_01_q2.mp4': {54.530535: 54.76, 84.999942: 84.56, 103.814954: 103.76, 111.146512: 111.2, 139.486694: 139.16, 144.726002: 144.52, 157.285727: 157.28, 195.579273: 195.6, 220.18514: 220.16, 234.265371: 234.24, 357.976437: 357.88}, '1552493137730_sc99_01_q3.mp4': {30.438019: 29.16, 47.225115: 46.08, 61.087326: 60.32, 112.24096: 110.8, 131.274274: 130.44, 163.972693: 162.16, 192.048804: 191.0, 233.349006: 232.04}, '1552493137730_sc99_01_q4.mp4': {5.260023: 4.2, 17.948099: 17.04, 65.092402: 64.16, 78.451006: 77.8, 113.673009: 111.68, 158.604245: 157.6, 165.033543: 163.96, 177.679326: 176.64, 247.594769: 246.52, 271.9916: 271.16}}

    writeNewJsonFileWithCorrectedTime(oldJsonFile, newJsonFile, correctedLabels)

def test_writeNewJsonFileWithCorrectedTime_2():
    oldJsonFile = '/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/test/extracted/label/Project_all_corrected.aucvl'
    
    newJsonFile = '/mnt/gpu02_raid/data/video/CBA/CBA_5_test_videos/test/extracted/label/Project_all_corrected_2.aucvl'
    
    # for testing videos
    correctedLabels = {'647b025243d74e719b36d24a0c19df37_sc99_q1.mp4': {19.111622: 19.96, 61.562372: 61.92, 104.782219: 104.36, 118.633067: 118.28, 206.170203: 207.08}, '647b025243d74e719b36d24a0c19df37_sc99_q2.mp4': {215.188944: 213.32}, '647b025243d74e719b36d24a0c19df37_sc99_q3.mp4': {15.191627: 12.92, 41.601645: 39.28, 64.855057: 62.84, 87.095742: 85.12, 117.040385: 115.12, 131.420537: 129.32, 174.541571: 172.36, 179.317041: 177.44, 185.457524: 183.32, 260.146794: 258.28}, '647b025243d74e719b36d24a0c19df37_sc99_q4.mp4': {6.644921: 5.68, 79.745543: 78.64, 85.065624: 84.0, 210.61878: 209.44, 221.958289: 220.96, 249.406368: 248.64, 263.446492: 262.52, 313.685375: 312.72, 318.785965: 317.72, 323.605373: 322.48, 329.505133: 328.6, 463.585691: 462.4, 493.980272: 493.12, 517.740188: 516.72}, '5102216385_5004650317_92_q1.mp4': {151.5593: 151.64, 167.530379: 167.12}, '5102216385_5004650317_92_q2.mp4': {18.228156: 17.44, 38.343704: 37.4, 85.143664: 84.28, 109.524643: 108.76, 129.340683: 128.48, 147.243998: 146.4, 177.324498: 176.32}, '5102216385_5004650317_92_q3.mp4': {}, '5102216385_5004650317_92_q4.mp4': {}, '1552141947911_sc99_01_q1.mp4': {29.399622: 28.6, 44.446105: 42.92, 73.458163: 72.08, 219.958483: 218.44, 240.137985: 239.4}, '1552141947911_sc99_01_q2.mp4': {1.111153: 0.8, 24.680341: 24.88, 83.519913: 83.32, 154.906784: 155.2, 166.772923: 166.44, 224.221812: 223.92, 235.681292: 235.44, 253.771892: 253.52, 273.207706: 272.76}, '1552141947911_sc99_01_q3.mp4': {11.173674: 10.84, 30.597816: 30.4, 47.120073: 47.0, 62.551377: 62.48, 106.380656: 106.08}, '1552141947911_sc99_01_q4.mp4': {200.266239: 199.12, 234.193276: 233.32, 331.444729: 330.04, 338.470053: 337.56}}

    writeNewJsonFileWithCorrectedTime(oldJsonFile, newJsonFile, correctedLabels)

if __name__ == "__main__":
    #testGetVideoAndEventLabels()
    #test_writeNewJsonFileWithCorrectedTime()
    test_writeNewJsonFileWithCorrectedTime_2()