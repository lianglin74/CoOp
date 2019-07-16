# GPU Cluster Job Management

## AML
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

2. Create the config file of `aux_data/configs/vigblob_account.yaml` for azure storage.
   The file format is
   ```yaml
   account_name: xxxx
   account_key: xxxx
   sas_token: ?xxxx
   container_name: xxxx
   ```
   Note, the SAS token should start with the question mark.

3. Create the config file of `aux_data/aml/config.json` to specify the
   subsription information
   ```json
   {
       "subscription_id": "xxxx",
       "resource_group": "xxxxx",
       "workspace_name": "xxxxx"
   }
   ```
   Note, leave the double quotes there to make it a valid json file.

4. Create the config file of `aux_data/aml/aml.yaml` to specify the submission
   related parameters. Here is one example.
   ```yaml
   azure_blob_config_file: ./aux_data/configs/vigblob_account.yaml
   # the name for the azure storage account, used as a tag in AML
   datastore_name: vig_data 
   # used to initialize the workspace
   aml_config: aux_data/aml/config.json 

   # the following is related with the job submission. If you don't use the
   # submission utility here, you can set any value

   # during job submission, aml-sdk will upload all data in this folder
   source_directory: ./src/qd/gpucluster
   # this is the entry point the AML will execute in the cluster
   entry_script: aml_server.py
   config_param: 
       # the path here is relative to the azure blob container
       # where the zipped source code is
       code_path: alias/code/quickdetection.zip 
       data_folder: alias/data/qd_data # after the source code is unzipped, this folder will be as $ROOT/data
       model_folder: alias/work/qd_models # this folder will be as $ROOT/models
       output_folder: alias/work/qd_output # this folder will be as $ROOT/output
   # if False, it will use AML's PyTorch estimator, which is not heavily tested here
   use_custom_docker: true
   # this is from AML admin. don't change it unless got notified
   compute_target: NC24RSV3 
   docker:
       # the custom docker. If use_custom_docker is False, this will be ignored
       image: amsword/setup:py36pt11 
   ```

5. Set an alias
   ```bash
   set a='ipython --pdb src/qd/gpucluster/aml_client.py -- '
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
   ```bash
   a submit cmd
   ```
   cmd is a string or a whitespace seperated string without quotes. What it
   does
   1. Upload all the source code in `aml.yaml:source_directory` to AML. Say, the
      folder in AML is ROOT_AML
   2. Make the current folder as `{ROOT_AML}` in AML
   3. Run the following command in AML
      ```bash
      python {aml.yaml:entry_script} \
          --code_path mount_point_for_config_param_code_path \
          --data_folder mount_point_for_config_param_data_folder \
          --model_folder mount_point_for_config_param_model_folder \
          --output_folder mount_point_for_config_param_output_folder \
          --cmd {cmd}
      ```
      If you are using the `aml_server.py` as the entry script. You can run any
      shell command by `a submit`. For example,
      - if you want to run `nvidia-smi` in
      AML. The command is
      ```bash
      a submit nvidia-smi
      ```
      - If you want to run `python train.py --data voc20` in AML, the command
      will be
      ```bash
      a submit python train.py --data voc20
      ```
      Note
      - By default, it uses 4 GPU with mpi as the distributed backend. That
      means, the command will be executed four times with different environment
      variables.
      - If you want to use 8 GPU, run the command like
      ```bash
      a -n 8 submit python train.py --data voc20
      ```
      `-n 8` should be placed before submit. Otherwise, it will think `-n 8` as
      part of the cmd


## Philly

