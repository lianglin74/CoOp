# Human Evaluation For Image Tagging and Detection
This folder contains scripts to do human evaluation for image tagging and detection.
Human evaluation is needed when the ground truth labels are not available or complete.
So human judges are asked to verify if the prediction results are correct or
not. Correct prediction will be added to ground truth labels. Then evaluation
can be done on the updated ground truth.

## Evaluation datasets
All dataset related files are stored at `vigdgx02:/raid/data/[DATASET_NAME]`.
The class labels are **case insensitive**.
### Image Tagging
- GettyImages2k
- linkedin1k
- MIT1K-GUID
- Top100Instagram-GUID
### Object Detection
- GettyImages2k_with_bb
- linkedin1k_with_bb
- MIT1K_with_bb
- Top100Instagram_with_bb

## Requirements
To run the scripts, firstly set up qd
```bash
cd "${QUICKDETECTION_ROOT}/src"
python setup.py build develop

```
Make sure that `${QUICKDETECTION_ROOT}/data` points to the data folder on
vigdgx02
```bash
ln -s /raid/data data
```
## Run scripts
**1. Submit a predict file to verify**
```bash
cd "${QUICKDETECTION_ROOT}"
python evaluation/db_task.py \
    --task submit \
    --type [choose from tagging/detection] \
    --dataset [choose from evaluation datasets] \
    --predict_file [file path]
```

**2. Download human judgment results and update ground truth files**
Before downloading, be sure to check if all the submitted tasks are finished.
For tagging, go to <https://prod.uhrs.playmsn.com/Manage/Task/TaskList?hitappid=35851>
For detection, go to <https://prod.uhrs.playmsn.com/Manage/Task/TaskList?hitAppId=35716&taskOption=0&project=-1&taskGroupId=-1>
```bash
cd "${QUICKDETECTION_ROOT}"
python evaluation/db_task.py \
    --task download \
    --type [choose from tagging/detection] \
    --dataset [choose from evaluation datasets]
```

