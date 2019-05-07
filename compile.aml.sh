#!/bin/bash
# don't continue if there is any error
set -e

# save the root folder so that we can always cd back
QUICKDETECTION_ROOT=$(pwd)

# compile faster-rcnn
PY_FASTER_RCNN_ROOT="${QUICKDETECTION_ROOT}/src/py-faster-rcnn"

cd "${PY_FASTER_RCNN_ROOT}/lib"
find . -name '*.so' -exec rm -rf {} \;
make

#sudo python -m nltk.downloader all

cd "${QUICKDETECTION_ROOT}/src"
python setup.py clean --all
python setup.py build develop

cd "${QUICKDETECTION_ROOT}/src/objectdetection"
python setup.py clean --all
python setup.py build develop

#compile the maskrcnn
cd ${QUICKDETECTION_ROOT}/src/maskrcnn-benchmark/
python setup.py build develop
