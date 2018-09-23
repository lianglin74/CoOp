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
2. install the dependency
   ```
   ./install_dep.sh
   ```

2. Compile the source code by
   ```
   ./compile.sh
   ```
3. (Optional) mkdir data/ and mkdir output. Link the source dataset to data and link your existing model folder to output. 

   Copy the data from //ivm-server2/IRIS/IRISObjectDetection/Data/datasets to QuickDetectionRoot/data. 
   Copy the model from //ivm-server2/IRIS/IRISObjectDetection/Data/imagenet_models to QuickDetectionRoot/models.

   Run the following to train the model, followed by testing
   ```
   python scripts/train.py --gpu 0 --net zf --data voc20 --iters 7000 --expid dbg
   ```

# Examples
## Taxonomy
1. create a full txonomy yaml file based on the hierarchical file and mapping file
   ```
   python scripts/tools.py -p \
       "{'type': 'create_taxonomy_based_on_vso', \
       'hier_tax_yaml': '/input_path/ObjectDetection/Taxonomy/Latest/taxonomy.yaml', \
       'property_yaml': '/input_path/ObjectDetection/Taxonomy/Latest/mapping.yaml', \
       'full_taxonomy_yaml': '/output_path/root.yaml'}"
   ``` 
2. extract the hier taxonomy yaml and the properties or mapping yaml from a full taxonomy yaml file
   ```
   python scripts/tools.py -p \
       "{'type': 'extract_full_taxonomy_to_vso_format', \
       'hier_tax_yaml': '/output_path/t.yaml', \
       'property_yaml': '/output_path/p.yaml', \
       'full_taxonomy_yaml': '/input_path/a.yaml'}"
   ```


## Training
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

## Visualization
1. Visualize the data under data/ and the model prediction result under output/
   1. go to the folder of visualization
      ```
      cd visualize
      ```
   2. Launch the django server
      ```
      python manage.py runserver 0:8000
      ```
   3. Go the website of http://ip:8000/detection/view_image to view the image
   4. Go to the website of http://ip:8000/detection/view_model to view all the
      model under output/
   5. Go to http://ip:8000/detection/view_tree/ to view the tree structure data

## Evaluation
1. Evaluate detection results
    * Copy all the files from `\\ivm-server2\IRIS\OD\eval\prediction` to `./groundtruth`.

    ```
    mkdir groundtruth && cd groundtruth
    cp -r path_to_groundtruth_folder/* .
    ```
    This folder includes baseline detection results on `MIT1k` and `Instagram` and their ground truth labels.

    * Run prediction using your own model, update ground truth labels if needed.
    See details at https://cognitionwiki.com/display/OB/How+to+evaluate+OD+models+via+human

    * Run evaluation
    ```
    python evaluation/human_eval.py --set mit1k --result PATH_TO_RESULT
    python evaluation/human_eval.py --set instagram --result PATH_TO_RESULT
    ```
    Or just run
    ```
    python evaluation/human_eval.py
    ```
    to see the current baselines.
    ::
## Contribute
TODO: Explain how other colleagues can contribute. 
