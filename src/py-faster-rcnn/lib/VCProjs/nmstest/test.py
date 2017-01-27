import json;
import numpy as np;
from nms_wrapper  import nms;

with open('test.tsv','r') as tsvin:
    rects  = json.load(tsvin);

nrect = len(rects);
dets = np.zeros((nrect,5), dtype=np.float32);

for i in range(nrect):
    dets[i,:4] = rects[i]['rect'];
    dets[i,4] = rects[i]['conf'];

keep = nms(dets, 0.3)
rectout = [ rects[i] for i in keep ]
with open('test.out',"w") as tsvout:
    json.dump(rectout,tsvout,indent=4);

