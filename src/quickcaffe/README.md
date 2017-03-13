
### Introduction
This repo is to provide python scripts for generating most common model structures (in prototxt), with the flexiblity of changing/modifying the input data layer and output loss layers.

The model structures provided in this repo include:  
1. CaffeNet (aka AlexNet with slight modification)  
2. ZF (CaffeNet variant to use input size 224)  
3. VGG (16 and 19 layers)  
4. GoogleNet (Inception V1)  
5. ResNet (10, 18, 34, 50, 101, 152 layers)  
6. ResNetRCNN (feature map resolution of 14x14 for the last stage)  
7. SqueezeNet ([v1.1](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1))

### Prerequisite
This repo assumes you have a caffe environment with `pycaffe` built successfully, either in Linux or Windows, and have a environment variable `$PYTHONPATH` pointing to the caffe root folder. To test if `pycaffe` works,
``` bash
$ python
>>>import caffe
```
If caffe can be successfully imported in python, you are good to go.

### Working Environment
Keep in mind you have three folders:  
1. This `quickcaffe` folder you synced from git. You can use `scripts/imagenet_benchmark.py` to run benchmark for ImageNet, or write your own script to do complex deep learning tasks. If your script is useful for other people, please contribute it using a Pull Request.  
2. The `data` folder as your working directory. Normally it includes big training files, or symbol links to big training files. Your net prototxt, solver prototxt, model snapshot, and log files should all be in this folder. Your should run `quickcaffe` script using an absolute path under this folder. For example  
``` bash
~/data$ python ~/git/quickcaffe/scripts/imagenet_benchmark.py --model ResNet18 --data train.tsv
```
This will generate prototxt files in this working directory, and then you can run `caffe` to start training.  
3. The `Caffe` root folder, which is a prerequisite to `quickcaffe`.  
