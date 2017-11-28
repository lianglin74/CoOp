# don't continue if there is any error
set -e

# save the root folder so that we can always cd back
QUICKDETECTION_ROOT=$(pwd)

# compile caffe
CAFFE_ROOT="${QUICKDETECTION_ROOT}/src/CCSCaffe/"
cd $CAFFE_ROOT
if [ ! -f 'Makefile.config' ]; then
    cp Makefile.config.example Makefile.config
    echo "USE_CUDNN := 1" >> Makefile.config
    echo "WITH_PYTHON_LAYER := 1" >> Makefile.config
    echo "USE_NCCL := 1" >> Makefile.config
    # under ubuntu, compilation will fail without c++11 option
    echo "CUSTOM_CXX := g++ --std=c++11" >> Makefile.config
fi
make proto
make -j
make pycaffe

# compile faster-rcnn
PY_FASTER_RCNN_ROOT="${QUICKDETECTION_ROOT}/src/py-faster-rcnn"
cd "${PY_FASTER_RCNN_ROOT}/lib"
make

