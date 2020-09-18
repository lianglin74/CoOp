# Human Evaluation For Image Tagging and Detection
This folder contains scripts to do human evaluation for image tagging and detection.
Human evaluation is needed when the ground truth labels are not available or complete.
Therefore, human judges are asked to verify if the prediction results are correct or
not. Correct prediction will be added to ground truth labels. Then evaluation
can be done on the updated ground truth.
[Here](https://prod.uhrs.playmsn.com/judge/Views/judge?HitAppID=35716&mode=pdesign&TaskgroupID=91381&TaskID=2634770&debug=1) is an example of the verification task. 

This approach is especially useful for comparing with competitors. There is no
need to get labels covering competitors' taxonomy. We can update ground truth
labels basing on competitors' prediction reuslts, and do the evaluation in an
open vocabulary manner.

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

## Get tagging/detection results from competitors

The competitors include Amazon Rekognition API, Google Could Vision API, and Clarifai.
To call the API, you need an account, set up authentication and
install SDKs. The documentations are provided:

- **Amazon**:

   Follow the [guide](https://docs.aws.amazon.com/rekognition/latest/dg/labels-detect-labels-image.html) to generate credential.
   Store credential in `~/.aws/`.

   Install the client:
   ```bash
   pip install boto3
   ```

- **Google**:

   Follow the [guide](https://cloud.google.com/vision/docs/detecting-objects) to
   generate the auth file.

   Before running the script, run
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=[PATH to auth file],
   ```
    or on Windows,
   ```bash
   set GOOGLE_APPLICATION_CREDENTIALS=[PATH to auth file]
   ```

   Install the client:
   ```bash
   pip install google-cloud-vision
   ```

- **Clarifai**:

   Follow the [guide](https://docs.clarifai.com/) to get API key.

   Install the client:
   ```bash
   pip install clarifai
   ```


After setting up environments, run the command to get prediction results
```bash
cd "${QUICKDETECTION_ROOT}"
python evaluation/call_api.py \
    --service [choose from microsoft, amazon, google, clarifai] \
    --target [choose from tag, detection, logo] \
    --dataset [choose from evaluation datasets] \
    --outfile [PATH]
```

## Update ground truth
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

The updated ground truth files will be saved in the according dataset folder.
The file path is printed on screen.
```bash
cd "${QUICKDETECTION_ROOT}"
python evaluation/db_task.py \
    --task download \
    --type [choose from tagging/detection] \
    --dataset [choose from evaluation datasets]
```

## Calculate Precision/Recall
After getting ground truth labels updated, Precision and Recall can be
calculated for all the models.

We use a YAML file to configure the evaluation settings, including ground truth
labels, baselines, and confidence thresholds. An example config file can be
found at `vigdgx02:/raid/data/GettyImages2k/api/config.yaml`.

With the config file, run the command to get a table of P/R of all baselines.
The P/R curve will be saved to the same directory as the config file.
```bash
cd "${QUICKDETECTION_ROOT}"
python evaluation/human_eval.py \
    --config [PATH to config file, e.g., /raid/data/GettyImages2k/api/config.yaml] \
    --iou_threshold [for detection results, default is 0.5] \
    --tag_only   # add this line for tagging evaluation
```

