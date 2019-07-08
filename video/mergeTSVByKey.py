try:
  from itertools import izip as zip
except:
  pass

import json
from qd import tsv_io
from qd.process_tsv import load_key_rects

from qd.tsv_io import tsv_reader, tsv_writer
from qd.qd_common import json_dump
import sys

def merge_by_key(tsv_file1, tsv_file2, out_tsv,
        from_flag1='', from_flag2=''):
    files = [tsv_file1, tsv_file2]
    
    all_key_rects = [load_key_rects(tsv_reader(f)) for f in files]
    
    
    all_key_to_rects = [ {key: rects for key, rects in key_rects}
        for key_rects in all_key_rects]

    keys = [key for key, _ in all_key_rects[0]]
    flags = [from_flag1, from_flag2]
    
    def gen_rows():
        for key in keys:
            all_rects = [key_to_rects[key] for key_to_rects in all_key_to_rects if key in key_to_rects]
            
            for rects, f in zip(all_rects, flags):
                for r in rects:
                    r['from'] = f
            rects = all_rects[0]
            for r in all_rects[1:]:
                rects.extend(r)
            yield key, json_dump(rects)
    tsv_writer(gen_rows(), out_tsv)


def main():

  tsv_file1 = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_demo_v2/1551538896210_sc99_01_q1.tsv"
  tsv_file2 = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_demo_v2/temp.tsv"

  out_tsv = "/mnt/gavin_ivm_server2_IRIS/ChinaMobile/Video/CBA/CBA_demo_v2/1551538896210_sc99_01_q1_withPerson.tsv"
  
  merge_by_key(tsv_file1, tsv_file2, out_tsv)

if __name__ == '__main__':
  #generateCBA_2_part01()
  if len(sys.argv) == 4:
    tsv_file1 = sys.argv[1]
    tsv_file2 = sys.argv[2]    
    out_tsv = sys.argv[3]
    
    merge_by_key(tsv_file1, tsv_file2, out_tsv)
  else:
    main()

