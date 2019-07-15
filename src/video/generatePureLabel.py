import json
from qd.tsv_io import tsv_reader
import sys


def generatePureLabel(tsvWithLabelFile, labelFileName):
    debug = 0

    f2 = open(labelFileName, 'w')

    lineCnt = 0

    t1 = tsv_reader(tsvWithLabelFile)

    for v1 in t1:
        lineCnt += 1
        if debug and (lineCnt > 4):
            exit()

        a = json.loads(v1[1])
        f2.write(v1[0] + "\t" + json.dumps(a) + "\n")

    f2.close()


def generateCBA_0():
    tsvWithLabelFile = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_0.withLabel.tsv"

    labelFileName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_0.pureLabel.tsv"
    generatePureLabel(tsvWithLabelFile, labelFileName)


def generateCBA_correct_0():
    tsvWithLabelFile = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/split_CBA_video_0/CBA_video_0.withLabel.correct.tsv"

    labelFileName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/split_CBA_video_0/CBA_video_0.pureLabel.correct.tsv"
    generatePureLabel(tsvWithLabelFile, labelFileName)


def generateCBA_1():
    tsvWithLabelFile = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_1.withLabel.tsv"

    labelFileName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_1.pureLabel.tsv"
    generatePureLabel(tsvWithLabelFile, labelFileName)


def generateCBA_correct_1():
    tsvWithLabelFile = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/split_CBA_video_1/CBA_video_1.withLabel.correct.tsv"

    labelFileName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/split_CBA_video_1/CBA_video_1.pureLabel.correct.tsv"
    generatePureLabel(tsvWithLabelFile, labelFileName)


def generateCBA_2_part01():
    tsvWithLabelFile = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_2_p01.withLabel.tsv"

    labelFileName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_2_p01.pureLabel.tsv"
    generatePureLabel(tsvWithLabelFile, labelFileName)


def generateCBA_2_part01_correct():
    tsvWithLabelFile = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_2_part01/CBA_video_2_p01.withLabel.correct.tsv"

    labelFileName = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_2_part01/CBA_video_2_p01.pureLabel.correct.tsv"
    generatePureLabel(tsvWithLabelFile, labelFileName)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        tsvWithLabelFile = sys.argv[1]
        labelFileName = tsvWithLabelFile.replace('withLabel', 'pureLabel')
        if labelFileName == tsvWithLabelFile:
            print('wrong label file')
            exit()
        generatePureLabel(tsvWithLabelFile, labelFileName)
    else:
        print("Please check arguments")
