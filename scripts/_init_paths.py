"""Set up paths for Fast R-CNN."""
import os.path as op
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

CODE_ROOT = op.join(op.dirname(op.dirname(op.realpath(__file__))),"src"); 
PYCAFFE_LIB = op.join(CODE_ROOT,'CCSCaffe','python')
FRCN_LIB = op.join(CODE_ROOT,'py-faster-rcnn','lib')
QUICKCAFFE_LIB = CODE_ROOT  # for quickcaffe
add_path(PYCAFFE_LIB)
add_path(FRCN_LIB)
add_path(QUICKCAFFE_LIB)
