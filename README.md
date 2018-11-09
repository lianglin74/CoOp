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
2. Install nccl2 if you have not done so

   cuda 9 (works with ubuntu 16.04)
   ```
    wget https://amsword.blob.core.windows.net/setup/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64-cuda9.deb -O a.deb
    sudo dpkg -i a.deb
    sudo apt-get install -y libnccl2=2.2.12-1+cuda9.0 libnccl-dev=2.2.12-1+cuda9.0
    rm a.deb
   ```

   cuda 9.2 (works with ubuntu 17.10 and ubuntu 16.04)
   ```
    wget https://amsword.blob.core.windows.net/setup/nccl-repo-ubuntu1604-2.3.5-ga-cuda9.2_1-1_amd64.deb -O a.deb
    sudo dpkg -i a.deb
    sudo apt install libnccl2=2.3.5-2+cuda9.2 libnccl-dev=2.3.5-2+cuda9.2
   ```

   [Check here to find which version you have installed for cuda](https://medium.com/@changrongko/nv-how-to-check-cuda-and-cudnn-version-e05aa21daf6c)

2. install the dependency
   ```
   ./install_dep.sh
   ```

3. Compile the source code by
   ```
   ./compile.sh
   ```

   1. if you find the following error. 
   ```
   error: use of enum ‘cudaFuncAttribute’ without previous declaration
   ```
   please update the file of cudnn.h by 
   sudo opening /usr/include/cudnn.h, and then changing the line of 
   ```
    #include "driver_types.h" 
   ```
   to 
   ```
    #include <driver_types.h>
   ```
   [See here for more context](https://devtalk.nvidia.com/default/topic/1025801/cudnn-test-did-not-pass/)

   2. if you find the error of internal compiler error: Killed (program cc1plus). Please compile the caffe with 
   fewer cpus by replacing make -j to make -j2 in compile.sh. 

   [See here for more context](https://stackoverflow.com/questions/30887143/make-j-8-g-internal-compiler-error-killed-program-cc1plus)

   3. if you find the folloowing error 
   ```
   classification.o: undefined reference to symbol '_ZN2cv6imreadERKNS_6StringEi'
   ```
   most likely, you have opencv 3 installed. Uncomment the following in
   src/CCSCaffe/Makefile.config
   ```
   # OPENCV_VERSION := 3
   ```
   [See here for more context](https://stackoverflow.com/questions/31253870/caffe-opencv-error)

   4. If you find the link error for opencv when using opencv 3
   replace the following line in Makefile
   ```
	ifeq ($(OPENCV_VERSION), 3)
		LIBRARIES += opencv_imgcodecs
	endif
   ```
   with
   ```
	ifeq ($(OPENCV_VERSION), 3)
		LIBRARIES += opencv_imgcodecs opencv_videoio
	endif
   ```


4. (Optional) mkdir data/ and mkdir output. Link the source dataset to data and link your existing model folder to output. 

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

## Predict
1. How to run a model against a data split
   1. Prepare
      1. Each model has a folder, e.g. 'voc20_darknet19_A'. We call it
         full_expid
      2. We use test_data and test_split to identify the test data. test_data
         is the folder name under data/. test_split is the data name. The file
         name should be like '{}/{}.tsv'.format(test_data, test_split). It is
         also feasible if it is a composite data split.
   2. Command line
      ```
      python scripts/tools.py -p "{'type': 'yolo_predict', \
            'full_expid': prepared_full_expid_name, \
            'test_data': prepared_test_data_name, \
            'test_split': prepared_test_split_name, \
            'gpus': [0,1,2,3]}"
      ```
      The output will be under 'output/{}/snapshot/'.format(full_expid)

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
2. Visualize comparsion between two models prediction results/
   1. link  predict result to ./data/compare
        eg: ln /mnt/ivm-server2_od/eval/prediction ./data/compare
   2. Go to http://$server_name:8000/detection/view_compare

## Deploy
1. Convert the model trained with -full_gpu to the model without -full_gpu
   1. Context: in the deployment env, we might still use the old structure,
      where we use RegionOutput layer. The new structure uses the
      RegionPrediction layer. The main difference is the layout of the last
      conv layer's output. The old one is x, y, w, h, obj, cls, x, y, w, h,
      obj, cls, e.t.c. The newer one is xxxxx, yyyy, wwwww. The initailization
      algorithm of the deployment env might copy the weight for the bounding
      boxes, i.e. x, y, w, h. Thus, we have to convert the model to the old
      version. 
   2. Command line: 
   ```
   python scripts/tools.py -p "{'type': 'convert_full_gpu_yolo_to_non_full_gpu_yolo', \
                                'full_expid': 'brand1048_darknet19_448_B_noreorg_rotate10_init3553_extraConvKernel.1.3.1_TsvBoxSamples50ExtraConvGroups1_4_1EffectBatchSize128'}"
   ```

## Evaluation
1. Human evaluation on MIT1K and Instagram
    1. Run detector on datasets, default is `MIT1k` and `Instagram`.
       Name the prediction result file as [dataset].[source].tsv,
       where source should be a unique, concise name to describe the model.

    2. Copy all the files from `\\ivm-server2\IRIS\OD\eval\prediction` to `./groundtruth`.
        ```
        mkdir groundtruth && cd groundtruth
        cp -r path_to_groundtruth_folder/* .
        ```
       This folder includes baseline detection results and their ground truth labels.

    3. If needed, update ground truth labels with new detection results,
		   see details at https://cognitionwiki.com/display/OB/How+to+evaluate+OD+models+via+human

       Copy honey pot labels from `\\ivm-server2\IRIS\OD\eval\honeypot`
        ```
        mkdir honeypot
        cp -r path_to_honeypot_folder/* honeypot/
        ```
       Compile C# code in evaluation/UHRSDataCollection. Run script
        ```
        mkdir tasks
        python evaluation/update_gt.py MODEL_SOURCE PATH_TO_RESULT_FOLDER \
            --dataset instagram mit1k \
            --gt ./groundtruth/config.yaml \
            --task ./tasks/ \
            --honeypot ./honeypot/voc20_easy_gt.txt
        ```

    4. Run evaluation
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

2. Evaluate a model trained on the composite dataset on all test sets 
    * Replace the full expid to adapt to your model
    ```
    python scripts/tools.py -p \
        "{'type': 'evaluate_tax_fullexpid', \
          'full_expid': 'Tax1300V14.1_0.0_0.0_darknet19_448_B_noreorg_bb_only'}"
    ```

3. Evaluate a model on the test split of voc20
    ```
    python scripts/tools.py -p \
        "{'type': 'yolo_predict', \
          'full_expid': 'Tax1300V14.1_0.0_0.0_darknet19_448_B_noreorg_bb_only', \
          'test_data': 'voc20', \
          'test_split': 'test'}"
    ```

## Contribute
TODO: Explain how other colleagues can contribute. 
