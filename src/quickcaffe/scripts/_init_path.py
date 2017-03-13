import os
import os.path as op
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# Add lib to PYTHONPATH, so that we can import modelzoo
this_dir = op.dirname(__file__)
lib_path = op.join(this_dir, '..')
add_path(lib_path)
