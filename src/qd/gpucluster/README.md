# GPU Cluster Job Management

## AML
### Installation
1. Download and install the source code
   * install with pip
     ```bash
     pip install "git+https://visionbio@dev.azure.com/visionbio/quickdetection/_git/quickdetection#egg=qd&subdirectory=src"
     ```
     if you have set up the ssh for authentication, you can run the following
     to install
     ```bash
     pip install "git+ssh://git@ssh.dev.azure.com/v3/visionbio/quickdetection/quickdetection#egg=qd&subdirectory=src"
     ```
   * or, install by download the source code explicitly
     ```bash
     git clone git@ssh.dev.azure.com:v3/visionbio/quickdetection/quickdetection
     cd src
     python setup.py build develop
     ```
   Note
   - This will automatically install the dependencies. If you don't want to
   install the dependencies, you can add `--no-deps` in `pip` command or
   `setup.py` command, where you need to manually install the dependency if
   some package is missing.
   - If you are using system-level python, please use the option of `--user` 
   in setup command so that the lib won't contaminate the system lib

2. Setup azcopy(optional)
   azcopy is used to upload or download data to azure blob. If the tool is only
   used for job management, then there is no need to set this up.

   Following [this link](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10)
   to download the azcopy and make sure the azcopy is downloaded to
   ~/code/azcopy/azcopy. That is, you can run the following to check if it is
   good.
   ```shell
   ~/code/azcopy/azcopy --version
   ```
   Make sure it is NOT version 8 or older.


3. Create the config file of `aux_data/configs/vigblob_account.yaml` for azure storage.
   The file format is
   ```yaml
   account_name: xxxx
   account_key: xxxx
   sas_token: ?xxxx
   container_name: xxxx
   ```
   Note, the SAS token should start with the question mark.

4. Create the config file of `aux_data/aml/config.json` to specify the
   subsription information
   ```json
   {
       "subscription_id": "xxxx",
       "resource_group": "xxxxx",
       "workspace_name": "xxxxx"
   }
   ```
   Note, leave the double quotes there to make it a valid json file.

5. Create the config file of `aux_data/aml/aml.yaml` to specify the submission
   related parameters. Here is one example.
   ```yaml
   azure_blob_config_file: null # no need to specify, legacy option
   datastore_name: null # no need to specify. legacy option
   # used to initialize the workspace
   aml_config: aux_data/aml/config.json 

   # the following is related with the job submission. If you don't use the
   # submission utility here, you can set any value

   config_param: 
      code_path:
          azure_blob_config_file: ./aux_data/configs/vigeastblob_account.yaml # the blob account information
          path: jianfw/code/quickdetection.zip # where the zipped source code is
      # you can add multiple key-value pairs to configure the folder mapping.
      # Locally, if the folder name is A, and you want A to be a blobfuse
      # folder in the AML side, you need to set the key as A_folder. For
      # example, if the local folder is datasets, and you want datasets to be a
      # blobfuse folder in AML running, then add a pair with the key being
      # datasets_folder.
      data_folder:
          azure_blob_config_file: ./aux_data/configs/vigeastblob_account.yaml # the blob account information
          # after the source code is unzipped, this folder will be as $ROOT/data
          path: jianfw/data/qd_data 
      output_folder:
          azure_blob_config_file: ./aux_data/configs/vigeastblob_account.yaml # the blob account information
          path: jianfw/work/qd_output # this folder will be as $ROOT/output
   # if False, it will use AML's PyTorch estimator, which is not heavily tested here
   use_custom_docker: true
   # this is from AML admin. don't change it unless got notified
   compute_target: NC24RSV3 
   # if it is the ITP cluster, please set it as true
   aks_compute: false
   docker:
       # the custom docker. If use_custom_docker is False, this will be ignored
       image: amsword/setup:py36pt16
   # any name to specify the experiment name.
   # better to have alias name as part of the experiment name since experiment
   # cannot be deleted and it is better to use fewer experiments
   experiment_name: your_alias 
   # if it is true, you need to run az login --use-device-code to authorize
   # before job submission. If you don't set it (default), it will prompt website to ask
   # you do the authentication. It is recommmended to set it as True
   use_cli_auth: True
   # if it is true, it will spawn n processes on each node. n equals #gpu on
   # the node. otherwise, there will be only 1 process on each node. In
   # distributed training, if it is false, you might need to spawn n extra
   # processes by yourself. It is recommended to set it as true (default)
   multi_process: True
   gpu_per_node: 4
   env:
      # the dictionary of env will be as extra environment variables for the
      # job running. you can add multiple env here. Note, sometimes the default
      # of NCCL_IB_DISABLE is '1', which will disable IB. highly recommneded to
      # alwasy set it as '0', even when IB is not available.
      NCCL_IB_DISABLE: '0'
   ```

6. Set an alias
   ```bash
   alias a='ipython --pdb src/qd/gpucluster/aml_client.py -- '
   ```
   or (when the current folder is not quickdetection)
   ```bash
   alis a='python -m qd.gpucluster.aml_client '
   ```
   or
   ```bash
   alis a='AML_CONFIG_PATH=path_to_config python -m qd.gpucluster.aml_client '
   ```

### Job Management
1. How to query the job status
   ```bash
   # the last parameter is the run id
   a query jianfw_1563257309_60ce2fc7
   # the last parameter can be the last several characters in the run id. it
   # will automatically match all the existing ones. If there is only one
   # matched, we will use that. Otherwise, it will crash
   a query e2fc7
   # replace query by q to save typing
   a q e2fc7
   ```
   What it does
   1. Download the logs to the folder of `./assets/{RunID}`
   2. Print the last 100 lines of the log for ranker 0 if there is.
   3. Print the log paths so that you can copy/paste to open the log
   4. Print the meta data about the job, including status.
   One example of the output is

   ```bash
   0.2594)  loss_objectness: 0.0500 (0.0625)  loss_rpn_box_reg: 0.0438 (0.0539)  time: 0.9798 (0.9946)  data: 0.0058 (0.0134)  lr: 0.020000  max mem: 3831
   2019-07-16 20:41:29,098.098 trainer.py:138   do_train(): eta: 13:02:24  iter: 42800  speed: 16.1 images/sec  loss: 0.4821 (0.4971)  loss_box_reg: 0.1157 (0.1214)  loss_classifier: 0.2480 (0.2593)  loss_objectness: 0.0545 (0.0625)  loss_rpn_box_reg: 0.0383 (0.0539)  time: 0.9876 (0.9946)  data: 0.0056 (0.0133)  lr: 0.020000  max mem: 3831
   2019-07-16 20:43:07,526.526 trainer.py:138   do_train(): eta: 13:00:43  iter: 42900  speed: 16.3 images/sec  loss: 0.4585 (0.4971)  loss_box_reg: 0.1045 (0.1214)  loss_classifier: 0.2289 (0.2593)  loss_objectness: 0.0551 (0.0625)  loss_rpn_box_reg: 0.0506 (0.0539)  time: 0.9807 (0.9946)  data: 0.0058 (0.0133)  lr: 0.020000  max mem: 3831
   2019-07-16 20:44:46,805.805 trainer.py:138   do_train(): eta: 12:59:03  iter: 43000  speed: 16.1 images/sec  loss: 0.4569 (0.4970)  loss_box_reg: 0.1180 (0.1214)  loss_classifier: 0.2291 (0.2592)  loss_objectness: 0.0479 (0.0625)  loss_rpn_box_reg: 0.0436 (0.0539)  time: 0.9802 (0.9946)  data: 0.0058 (0.0133)  lr: 0.020000  max mem: 3831
   2019-07-16 14:30:26,592.592 aml_client.py:147      query(): log files:
   ['ROOT/assets/jianfw_1563257309_60ce2fc7/azureml-logs/70_driver_log_rank_0.txt',
    'ROOT/assets/jianfw_1563257309_60ce2fc7/azureml-logs/70_driver_log_rank_2.txt',
    ...
    'ROOT/assets/jianfw_1563257309_60ce2fc7/azureml-logs/55_batchai_execution-tvmps_e967edcdb10dd5e65827d221af1f6b246bb7d854790e27d26a677f78efe897ae_d.txt',
    'ROOT/assets/jianfw_1563257309_60ce2fc7/azureml-logs/55_batchai_stdout-job_prep-tvmps_e967edcdb10dd5e65827d221af1f6b246bb7d854790e27d26a677f78efe897ae_d.txt',
    'ROOT/assets/jianfw_1563257309_60ce2fc7/azureml-logs/55_batchai_stdout-job_prep-tvmps_3bbfd76728dd63d173c5cb80221dc4b244254a0fd864c695c8e70bf9460ac7ae_d.txt']
   2019-07-16 14:30:27,096.096 aml_client.py:38 print_run_info(): {'appID': 'jianfw_1563257309_60ce2fc7',
    'appID-s': 'e2fc7',
    'cluster': 'aml',
    'cmd': 'python src/qd/pipeline.py -bp '
           'YWxsX3Rlc3RfZGF0YToKLSB0ZXN0X2RhdGE6IGNvY28yMDE3RnVsbAogIHRlc3Rfc3BsaXQ6IHRlc3QKcGFyYW06CiAgSU5QVVQ6CiAgICBGSVhFRF9TSVpFX0FVRzoKICAgICAgUkFORE9NX1NDQUxFX01BWDogMS41CiAgICAgIFJBTkRPTV9TQ0FMRV9NSU46IDEuMAogICAgVVNFX0ZJWEVEX1NJWkVfQVVHTUVOVEFUSU9OOiB0cnVlCiAgTU9ERUw6CiAgICBGUE46CiAgICAgIFVTRV9HTjogdHJ1ZQogICAgUk9JX0JPWF9IRUFEOgogICAgICBVU0VfR046IHRydWUKICAgIFJQTjoKICAgICAgVVNFX0JOOiB0cnVlCiAgYmFzZV9scjogMC4wMgogIGRhdGE6IGNvY28yMDE3RnVsbAogIGRpc3RfdXJsX3RjcF9wb3J0OiAyMjkyMQogIGVmZmVjdGl2ZV9iYXRjaF9zaXplOiAxNgogIGV2YWx1YXRlX21ldGhvZDogY29jb19ib3gKICBleHBpZDogTV9CUzE2X01heEl0ZXI5MDAwMF9MUjAuMDJfU2NhbGVNYXgxLjVfRnBuR05fRlNpemVfUnBuQk5fSGVhZEdOX1N5bmNCTgogIGV4cGlkX3ByZWZpeDogTQogIGxvZ19zdGVwOiAxMDAKICBtYXhfaXRlcjogOTAwMDAKICBuZXQ6IGUyZV9mYXN0ZXJfcmNubl9SXzUwX0ZQTl8xeF90YmFzZQogIHBpcGVsaW5lX3R5cGU6IE1hc2tSQ05OUGlwZWxpbmUKICBzeW5jX2JuOiB0cnVlCiAgdGVzdF9kYXRhOiBjb2NvMjAxN0Z1bGwKICB0ZXN0X3NwbGl0OiB0ZXN0CiAgdGVzdF92ZXJzaW9uOiAwCnR5cGU6IHBpcGVsaW5lX3RyYWluX2V2YWxfbXVsdGkK',
    'elapsedTime': 15.27,
    'num_gpu': 8,
    'start_time': '2019-07-16T06:14:10.688519Z',
    'status': 'Canceled'}
   ```

2. How to abort/cancel a submitted job
   ```bash
   a abort jianfw_1563257309_60ce2fc7
   a abort 60ce2fc7
   ```

3. How to resubmit a job
   ```bash
   a resubmit jianfw_1563257309_60ce2fc7
   a resubmit 60ce2fc7
   ```

4. How to submit the job
   The first step is to upload the code to azure blob by running the following
   command
   ```bash
   a init
   ```
   Whenever you want your new code change took effect, you should run the above
   command.
   To execute a command in AML, run the following:
   ```bash
   a submit cmd
   ```
   - if you want to run `nvidia-smi` in AML. The command is
   ```bash
   a submit nvidia-smi
   ```
   - If you want to run `python train.py --data voc20` in AML, the command
   will be
   ```bash
   a submit python train.py --data voc20
   ```
   Note
   - If you want to use 8 GPU, run the command like
   ```bash
   a -n 8 submit python train.py --data voc20
   ```
   `-n 8` should be placed before submit. Otherwise, it will think `-n 8` as
   part of the cmd
   - If `multi_process=true`, effectively it runs `mpirun --hostfile hostfile_contain_N_node_ips --npernode gpu_per_node cmd`
       - the number of nodes x gpu_per_node == the number of gpu requested
       - highly recommended for distributed training/inference
   - If `multi_process=false`, effectively it runs `mpirun --hostfile hostfile_contain_N_node_ips --npernode 1 cmd`
       - still, the number of nodes x gpu_per_node == the number of gpu requested
   - For pytorch code, we normally need to run `init_process_group` function.
     One recommended way is to insert the following code of 
       ```
       from qd.torch_common import ensure_init_process_group
       ensure_init_process_group()
       ```
       This function works on AML and locally (if launching with mpirun) as
       well. On the local machine, it is also recommended to launch the
       distributed training by `mpirun -n 8 python train.py` rather than
       `python -m torch.distributed.launch train.py`.

## Philly
### Installation
1. Download and install the source code
   ```bash
   git clone git@ssh.dev.azure.com:v3/visionbio/quickdetection/quickdetection
   cd src
   python setup.py build develop
   ```
   Note
   - you don't have to use the option of `--recursive` to download all
   submodules. 
   - Recommended to build it with develop option so that you can
   modify the code and there is no need to re-compile it. 
   - If you are using system-level python, please use the option of `--user` 
   in setup command so that the lib won't contaminate the system lib
   - This is the same as AML tool.

2. Setup azcopy as for AML.

3. Create the configuration file of `aux_data/configs/multi_philly_vc.yaml`,
   which tells the cluster names we have.
   ```
    clusters:
        - wu1
        - sc2
   ```
   If you just want to focus on sc2, then remove `-wu1`. The VC name is
   hard-coded as `input` since there is no need to change it in our group.

4. Create the configuration file of `aux_data/configs/philly_vc.yaml`. The
   following is an example.
   ```
    vc: input
    cluster: wu1
    #cluster: sc2
    user_name: jianfw
    password: null
    blob_mount_point: /blob
    azure_blob_config_file: ./aux_data/configs/vigeastblob_account.yaml
    multi_process: true
    config_param:
        # please make sure each path starts with blob_mount_point
        #code_path: /hdfs/input/jianfw/code/quickdetection.zip
        code_path: /blob/jianfw/code/quickdetection.zip
        data_folder: /blob/jianfw/data/qd_data
        #data_folder: /hdfs/input/jianfw/data/qd_data
        model_folder: /blob/jianfw/work/qd_models
        output_folder: /blob/jianfw/work/qd_output
    docker:
        image: philly/jobs/test/vig-qd-env
        tag: py36ptnight
   ```
   - `vc`: this is the name of virtual cluster. In our group, it is input
   - `cluster`: wu1 or sc2. This value will be overwritten by the config file of
   `multi_philly_vc.yaml`
   - `user_name`: the credential information for philly job submission
   - `password`: if it is null, it will try to get the password from the env
   variable of `PHILLY_PASSWORD`. It is suggested to set the password in the
   env variable.
   - `azure_blob_config_file`: the config file for azure blob account, whose
   format is the same as AML's
   - `multi_process`: if it is true, it wil launch the script multiple times
   (the number of GPU times) and give each process different environement
   variables so that you know which GPU to use. It is essentially to launch the
   command with mpirun. If it is false, it will launch your command once.
   - `config_param`: tells where to find the code, data, output folder. If you
   want to use hdfs system to keep your data, specify the path started with
   /hdfs.

   - This example only uses one blob for data access. If we want to use
     multiple blobs, you can specify all blob information in blob_mount_info
     ```
      vc: input
      cluster: wu1
      #cluster: sc2
      user_name: jianfw
      password: null
      blob_mount_info:
          - blob_mount_point: /blob
            azure_blob_config_file: ./aux_data/configs/vigeastblob_account.yaml
            blob_fuse_options:
                - '-o'
                - "attr_timeout=240"
                - "-o"
                - "entry_timeout=240"
                - "-o"
                - "negative_timeout=120"
                - "--log-level=LOG_WARNING"
                - "-o"
                - "allow_other"
                - "--file-cache-timeout-in-seconds=10000000"
          - blob_mount_point: /data_blob
            azure_blob_config_file: ./aux_data/configs/vigblob_account.yaml
            blob_fuse_options:
                - '-o'
                - 'ro' # read-only
                - '-o'
                - "attr_timeout=240"
                - "-o"
                - "entry_timeout=240"
                - "-o"
                - "negative_timeout=120"
                - "--log-level=LOG_WARNING"
                - "-o"
                - "allow_other"
                - "--file-cache-timeout-in-seconds=10000000"
      multi_process: true
      config_param:
          #code_path: /hdfs/input/jianfw/code/quickdetection.zip
          code_path: /blob/jianfw/code/quickdetection.zip
          data_folder: /data_blob/jianfw/data/qd_data
          #data_folder: /hdfs/input/jianfw/data/qd_data
          model_folder: /blob/jianfw/work/qd_models
          output_folder: /blob/jianfw/work/qd_output
      docker:
          image: philly/jobs/test/vig-qd-env
          tag: py36ptnight
     ```


5. Set an alias
   ```bash
   alis p='ipython --pdb src/qd/gpucluster/philly_client.py -- '
   ```

### Job Management
1. How to query the job status
   ```shell
   p query _6873
   p q _6873
   ```
   The last word is the suffix of the job id. You can specify full name or
   partial name. One sample output is
   ```
   ...
   2019-09-09 15:45:13,577.577 qd_common.py:406    cmd_run(): finished the cmd run
   2019-09-09 15:45:13,579.579 philly_client.py:661 track_job_once(): satus = Running
   2019-09-09 15:45:13,579.579 philly_client.py:663 track_job_once(): ssh -tt jianfw@sshproxy.sc2.philly.selfhost.corp.microsoft.com -p 2200 ssh -tt -o StrictHostKeyChecking=no jianfw@10.0.0.71 -p 2234 -i /var/storage/shared/input/sys/jobs/application_1564705084178_6873/.ssh/id_rsa
   2019-09-09 15:45:13,580.580 philly_client.py:666 track_job_once(): https://philly/#/job/sc2/input/1564705084178_6873
   ```

2. How to abort the job
   ```
   p abort _6873
   ```

3. How to resubmit the job
   ```
   p resubmit _6873
   ```

4. How to ssh into an exsiting job
   ```shell
   p ssh application_1562349962206_12410
   ```

5. How to submit a job
   1. Do initialization first by
      ```shell
      p init
      ```
      which will compress the code in the current folder to a zip file and
      upload it to the cloud as specified in `philly_vc.yaml`.
   2. submit the job
      1. How to submit a job for SSH
         ```shell
         p submit ssh
         ```
         By default, it applies 4 GPU. If you want more, run
         ```shell
         p -n 8 submit ssh
         ```
         Note, `-n 8` should be placed before `submit`
      2. How to run `nvidia-smi`
         ```shell
         p submit nvidia-smi
         ```
      4. How to run `python scripts/train.py`
         ```shell
         p submit python scripts/train.py
         ```


