import os, os.path as op
import sys;

filein = sys.argv[1];
idxout = op.splitext(filein)[0]+".lineidx";

with open(filein,'r') as tsvin, open(idxout,'w') as tsvout:
    fsize = os.fstat(tsvin.fileno()).st_size
    fpos = 0;
    while fpos!=fsize:
	tsvout.write(str(fpos)+"\n");
        tsvin.readline();
        fpos = tsvin.tell();

'''#test
with open(idxout,"r") as idxin:
    idxarray = [int(x) for x in idxin.readlines() if x.strip()!=""]
    print(idxarray)

with open(filein,'r') as tsvin:
    for idx in idxarray:
        tsvin.seek(idx)
        print(tsvin.readline())
'''
