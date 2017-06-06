#!python3
import sys;
import os,os.path as op
import numpy as np
import argparse
import json;
import base64;
import cv2;
import hashlib

def murl_to_fname(murl) :
    hash = hashlib.md5()
    hash.update(murl)    
    return base64.b64encode(hash.digest()).decode().replace("/","_").replace("+","-").replace("="," ")+".jpg";
    
def decode_buf( jpgbuf ):
    nparr = np.fromstring(jpgbuf, np.uint8)
    try:
        return cv2.imdecode(nparr, -1);
    except:
        return None;
def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    return decode_buf(jpgbytestring);

def load_results(res_file):
    res = dict();
    with open(res_file,"r") as tsv_in:
        for line in tsv_in:
            cols = [x.strip() for x in line.split("\t")];
            if len(cols)==2:
                key = cols[0];
                res_str = cols[1] if cols[1]!="" else "[]";
                crects = json.loads(res_str);
                if len(crects)>0:
                    res[key]=crects;
    return res;


def visualize (img, truths, crects, threshold=0.9):
    for truth in truths:
        rect  = [int(x) for x in truth['rect']];
        cname = truth['class'];
        cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0),2);
        cv2.putText(img,cname, (max(0,rect[0]-10),max(0, rect[1]-10)),cv2.FONT_HERSHEY_SIMPLEX, 0.5 ,(255,0,0),2);    
    for crect in crects:
        rect  = [int(x) for x in crect['rect']];
        cname = crect['class'];
        score = crect['conf'];
        if score<threshold:
            continue;
        cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),(0,0,255),2);
        cv2.putText(img,'{:s} {:.3f}'.format(cname,score ), (max(0, rect[0]-10),(rect[1]+rect[3])/2),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2);
    return img
        
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object Detection evaluation')
    parser.add_argument('-t', '--truth', required=True,   help='import groundtruth')
    parser.add_argument('-d', '--dets', required=False, default='',  help='import detection results')
    parser.add_argument('-c', '--confth', required=False, type=float, default=0.9, help="bbox confidence threshold")
    parser.add_argument('-f', '--folder_out', required=True,  help="output folder name")
    
    args = parser.parse_args()
    return args  
            
def drawresult(truthfile, dets, threshold, folder_output):
    retdict = dict();
    with open(truthfile, "r") as tsvin:
        for line in tsvin:
            cols = [x.strip() for x in line.split("\t")]
            if len(cols)<2:
                continue;
            murl = cols[0]
            fname = murl_to_fname(murl)
            truths = json.loads(cols[1]);
            img = img_from_base64(cols[2]);
            crects = dets[murl] if murl in dets else [];
            outimg = visualize(img, truths, crects,threshold)
            cnames = set([truth['class'] for truth in truths])
            for cname in cnames:
                cfolder = op.join(folder_output, cname.replace("'",'_'))
                if not op.exists(cfolder):
                    os.makedirs(cfolder)
                cv2.imwrite(op.join(cfolder,fname), outimg); 
    
if __name__ == "__main__":  
    args = parse_args();
    dets = load_results(args.dets);
    drawresult(args.truth, dets, args.confth, args.folder_out);
