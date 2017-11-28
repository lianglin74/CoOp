# IRIS Object Detection
## Introduction 
This repo is for the algorithm development for IRIS object detection. The current implementation is mainly based on `caffe` and `py-faster-rcnn`.

# Windows

## Prerequisite

1.	NVidia Cuda 8.0
2.	Visual Studio 2015
3.  [Anaconda for Python 2.7](https://repo.continuum.io/archive/Anaconda2-4.3.0.1-Windows-x86_64.exe)
## Installation
1. Clone the repository:

   ```
   git clone --recursive https://msresearch.visualstudio.com/DefaultCollection/IRIS/_git/IRISObjectDetection
   ```
2. Setup `Caffe` under the repo root folder:
   ```
   python scripts\setup_caffe.py
   ```
   The script will download 3rd party dependencies and build caffe automatically.  
   See [Caffe README.md](https://msresearch.visualstudio.com/IRIS/_git/CCSCaffe?path=%2FREADME.md&version=GBWinCaffe&_a=contents) for more detailed steps.
3. Setup `py-faster-rcnn` under the repo root folder (**Admin privileges is required**)
   ```
   python scripts\setup_pyfrcn.py
   ```
   The script will install required python packages. it will also  download test data and test model automatically.
   The test data sets contains `voc20` and `voc2`. `voc20` is a voc full dataset in TSV format, and `voc2` is a voc subset including only images for 'horse' and 'dog'.
   If you want to create other datasets, please refer `script\prepare_voc.py` for example.
   See [py-faster-rcnn README.md](https://msresearch.visualstudio.com/IRIS/_git/IRISObjectDetection?path=%2Fsrc%2Fpy-faster-rcnn%2FREADME.md&version=GBmaster&_a=contents) for more detailed steps.
4. Running examples under the repo root folder:
   ```
   python scripts\train.py --gpu 0 --net zf --data voc20 --iters 7000 --expid dbg
   ```

# Linux

## Installation
1. Clone the repository
   ```
   git clone --recursive https://github.com/leizhangcn/quickdetection.git 
   ```
2. Compile the source code by
   ```
   ./compile.sh
   ```
3. (Optional) Download the data and imagenet models. Run the training with testing.

   Copy the data from //ivm-server2/IRIS/IRISObjectDetection/Data/datasets to QuickDetectionRoot/data. 
   Copy the model from //ivm-server2/IRIS/IRISObjectDetection/Data/imagenet_models to QuickDetectionRoot/models.

   Note: currently, you have to copy the data from the network share to the
   local windows desktop/laptop and then upload it to the linux server.

   Run the following to train the model, followed by testing
   ```
   python scripts/train.py --gpu 0 --net zf --data voc20 --iters 7000 --expid dbg
   ```

## Examples
1. Run Yolo with a customized taxonomy
   1. Prepare the data. 
      1. Create a sub folder, named 'data' under $QD_ROOT (the root folder of
         this quickdetection folder)
      2. Link the data of voc20, coco2017, e.t.c. under data. 
         If you are working on DL cluster, the data are at /glusterfs/public/data, and you
         can link it by 
         ```
         ln -s /glusterfs/public/data/* data/
         ```
         If it is vig-gpu01, the data are located at /gpu02_raid/data/.
         If it is vig-gpu02, the data are located at /raid/data.
         If it is Windows, the data can be accessed by \\\\vig-gpu02\raid_data.
    2. Download the taxonomy data from https://github.com/leizhangcn/taxonomy10k
    3. Train the model with an extra parameter of --taxonomy_folder. One
       example is:
       ```
       python scripts/yolotrain.py --data office_v2.1 \
                --iters 128e \
                --expid 789 \
                --net darknet19 \
                --gpus 4 5 6 7 \
                --taxonomy_folder the_path_to_taxonomy
       ```
       Replace the_path_to_taxonomy by the path of the taxonomy folder

    
## Contribute
TODO: Explain how other colleagues can contribute. 
