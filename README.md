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
2. Visualize comparsion between two models prediction results/
   1. make folder under data to contain visualiztion result
   2. using python script build_compare.py under ./script with following
      parameters
      1. left side tsv filename
      2. right side tsv filename
      3. images tsv filename
      4. ouput tsv filenname(test.tsv)
      5. left side threshold filename
      6. right side threshold filename
      7. left side result tag
      8. right side result tag
      9. left side min confidence
      10. right side min confidence
   3. build visualization index 
      python ./scripts/process_tsv.py --type build_data_index --input
      $data_folder_name
   4. go to http://$server_name:8000/detection/view_compare and "select data folder"

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
