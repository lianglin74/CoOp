import subprocess32 as sp
import json

import os


def splitTSVs():
    tsvFiles = ["647b025243d74e719b36d24a0c19df37_sc99_.tsv", "1551538896210_sc99_01.tsv",
                "5102216385_5004650317_92.tsv", "1552141947911_sc99_01.tsv", "1552493137730_sc99_01.tsv"]

    secondsPerFile = 35*60
    framesInFile = secondsPerFile*25

    for fileName in tsvFiles:
        videoId = fileName.split('.')[0]
        newFileNamePrefix = videoId + "-s"

        cmd = "split --numeric-suffixes=1 -l " + \
            str(framesInFile) + "  " + fileName + " " + newFileNamePrefix
        os.system(cmd)
        cmd = "rename 's/-s0/-s/' " + newFileNamePrefix + "*"
        os.system(cmd)
        cmd = "ls *-s[1-9] | xargs -I fileName mv fileName fileName.tsv"
        os.system(cmd)


if __name__ == '__main__':
    splitTSVs()
