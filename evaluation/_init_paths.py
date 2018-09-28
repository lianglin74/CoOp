import os.path as op
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

QD_ROOT = op.dirname(op.dirname(op.realpath(__file__)))
add_path(op.join(QD_ROOT, "scripts"))
