# IRIS Object Detection
## Introduction 
This repo is for the algorithm development for IRIS object detection. The current implementation is mainly based on `caffe` and `py-faster-rcnn`.

## Prerequisite

1.	NVidia Cuda 8.0
2.	Visual Studio 2015

## Installation
1. Clone the repository:

   ```
   git clone --recursive https://msresearch.visualstudio.com/DefaultCollection/IRIS/_git/IRISObjectDetection
   ```
2. Setup `Caffe` by following [Caffe README.md](https://msresearch.visualstudio.com/IRIS/_git/CCSCaffe?path=%2FREADME.md&version=GBWinCaffe&_a=contents)
3. Under the repo root folder, copy `Pascal VOC' data:
   ```
   robocopy \\ivm-server2\IRIS\IRISObjectDetection\Data\voc2 data\voc2 /e
   robocopy \\ivm-server2\IRIS\IRISObjectDetection\Data\voc20 data\voc20 /e
   ```
   voc20 is a voc full dataset in TSV format, and voc2 is a voc subset including only images for 'horse' and 'dog'.

   If you want to create other datasets, please refer `script\prepare_voc.py` for example.
    
## Contribute
TODO: Explain how other colleagues can contribute. 
