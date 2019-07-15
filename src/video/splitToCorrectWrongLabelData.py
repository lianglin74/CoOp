try:
    from itertools import izip as zip
except:
    pass

import json
from qd import tsv_io
from qd.tsv_io import *


def splitCorrectWrongLabels(tsvOrgFile, labelFileName, correctLabelFileName, wrongLabelFileName):
    debug = 0

    fc = open(correctLabelFileName, 'w')
    fw = open(wrongLabelFileName, 'w')

    lineCnt = 0

    t1 = tsv_reader(labelFileName)
    t2 = tsv_reader(tsvOrgFile)

    keyToStateMap = {}
    for v1 in t1:
        keyToStateMap[v1[1]] = v1[2]

    for v2 in t2:
        lineCnt += 1
        if debug and (lineCnt > 16):
            exit()

        if v2[0] in keyToStateMap and keyToStateMap[v2[0]] == "No":
            a = json.loads(v2[1])
            fc.write(v2[0] + "\t" + json.dumps(a) + "\t" + v2[2]+"\n")
        else:
            a = json.loads(v2[1])
            fw.write(v2[0] + "\t" + json.dumps(a) + "\t" + v2[2]+"\n")

    fc.close()
    fw.close()


def workOnCBA_0_c():
    dir = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/split_CBA_video_0/c-yijuan/"
    tsvOrgFile = dir + "CBA_video_0.withLabel.c.tsv"
    labelFileName = dir + "Label/CBA_video_0.withLabel.c-Label-Dummy.txt"

    correctLabelFileName = dir + "c-CBA_video_0.withLabel.correct.tsv"
    wrongLabelFileName = dir + "c-CBA_video_0.withLabel.wrong.tsv"

    splitCorrectWrongLabels(tsvOrgFile, labelFileName,
                            correctLabelFileName, wrongLabelFileName)


def workOnCBA_0_b():
    dir = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/split_CBA_video_0/b-chunshui/"
    tsvOrgFile = dir + "CBA_video_0.withLabel.b.tsv"
    labelFileName = dir + "Label/CBA_video_0.withLabel.b-Label-Dummy.txt"

    correctLabelFileName = dir + "b-CBA_video_0.withLabel.correct.tsv"
    wrongLabelFileName = dir + "b-CBA_video_0.withLabel.wrong.tsv"

    splitCorrectWrongLabels(tsvOrgFile, labelFileName,
                            correctLabelFileName, wrongLabelFileName)


def workOnCBA_0_a():
    dir = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/split_CBA_video_0/a-gavin/"
    tsvOrgFile = dir + "CBA_video_0.withLabel.a.tsv"
    labelFileName = dir + "Label/CBA_video_0.withLabel.a-Label-Dummy.txt"

    correctLabelFileName = dir + "a-CBA_video_0.withLabel.correct.tsv"
    wrongLabelFileName = dir + "a-CBA_video_0.withLabel.wrong.tsv"

    splitCorrectWrongLabels(tsvOrgFile, labelFileName,
                            correctLabelFileName, wrongLabelFileName)


def workOnCBA_1_a():
    dir = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/split_CBA_video_1/a-yijuan/"
    tsvOrgFile = dir + "a-CBA_video_1.withLabel.tsv"
    labelFileName = dir + "Label/a-CBA_video_1.withLabel-Label-Dummy.txt"

    correctLabelFileName = dir + "a-CBA_video_1.withLabel.correct.tsv"
    wrongLabelFileName = dir + "a-CBA_video_0.withLabel.wrong.tsv"

    splitCorrectWrongLabels(tsvOrgFile, labelFileName,
                            correctLabelFileName, wrongLabelFileName)


def workOnCBA_1_b():
    dir = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/split_CBA_video_1/b-gavin/"
    tsvOrgFile = dir + "b-CBA_video_1.withLabel.tsv"
    labelFileName = dir + "Label/b-CBA_video_1.withLabel-Label-Dummy.txt"

    correctLabelFileName = dir + "b-CBA_video_1.withLabel.correct.tsv"
    wrongLabelFileName = dir + "b-CBA_video_0.withLabel.wrong.tsv"

    splitCorrectWrongLabels(tsvOrgFile, labelFileName,
                            correctLabelFileName, wrongLabelFileName)


def workOnCBA_2_part01():
    dir = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_2_part01/"
    tsvOrgFile = dir + "CBA_video_2_p01.withLabel.tsv"
    labelFileName = dir + "Label/CBA_video_2_p01.withLabel-Label-Dummy.txt"

    correctLabelFileName = dir + "CBA_video_2_p01.withLabel.correct.tsv"
    wrongLabelFileName = dir + "CBA_video_2_p01.withLabel.wrong.tsv"

    splitCorrectWrongLabels(tsvOrgFile, labelFileName,
                            correctLabelFileName, wrongLabelFileName)


def workOnCBA_2_part23():
    dir = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame/CBA_video_2_part23/"
    tsvOrgFile = dir + "toVerify_CBA_video_2_p23.withLabel.tsv"
    labelFileName = dir + "Label/toVerify_CBA_video_2_p23.withLabel-Label-Dummy.txt"

    correctLabelFileName = dir + "toVerify_CBA_video_2_p23.withLabel.correct.tsv"
    wrongLabelFileName = dir + "toVerify_CBA_video_2_p23.withLabel.wrong.tsv"

    splitCorrectWrongLabels(tsvOrgFile, labelFileName,
                            correctLabelFileName, wrongLabelFileName)


if __name__ == '__main__':
    workOnCBA_2_part23()
