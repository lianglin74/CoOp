#!/bin/bash

#### Note that this file was used to create a simple pipeline to call the scripts to generate TSV from UHRS vendor labeling results for the basketball video project. 
#### Now this needs to be changed to work since many scripts have to be moved to new folders and their names are also different. 
#### The purpose of checking in this file is to provide a template/example to invoke the child scripts to generate TSV files from UHRS vendor labeling results. 

runDir="/home/yaowe/dev/quickdetection"
cd $runDir

# !!maually!! move and clean the old files to processed folder

# !!maually!! download: Need to run on windows
#get the task id from upload folder to a file. 

# get the file list for result id
downloadDir="/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/uhrs/download"
fileListName="$downloadDir/filelist.txt"
echo "generating filelist for downloaded result files"
ls $downloadDir/[0-9]* -1 > $fileListName

read -p "Press enter to continue"

# 1. generate result
#!!!!!!!!!! to change !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
orgLabelResultWithDontlabel='CBA_video_2_p23.label.tsv'
map_file_name="/raid/data/ChinaMobileVideoCBA_video_2/train.key.url.tsv"

echo "----generating label file with dont label"
python generateTSVFromHURS.py $map_file_name $fileListName $orgLabelResultWithDontlabel

read -p "Press enter to continue"

# 2. python mergeTSV.py
#!!!!!!!!!! to change !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
topDataDir="/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/TSV_1frame"
tsvOrgFile="$topDataDir/CBA_video_2.tsv"
labelFileName="$runDir/$orgLabelResultWithDontlabel"
mergedFileName="$topDataDir/CBA_video_2_p23.withLabel.tsv"

echo "----generating tsv with label after removing dont label"
python mergeTSV.py $tsvOrgFile  $labelFileName $mergedFileName

read -p "Press enter to continue"

# 3. Split the TSV to get pure label
echo "----generating pure label for verification purpose"
python splitTSV.py $mergedFileName

read -p "Press enter to continue"

# manually verify

#...