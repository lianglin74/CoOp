#!python2
import os.path as op
import _init_paths
# import sys
# sys.path.insert(0, r'd:\github\caffe-msrccs\pythond')
import numpy as np
import caffe, os, sys, cv2
import argparse
import numpy as np
import base64
import progressbar 
import json
import matplotlib.pyplot as plt
import multiprocessing as mp
import glob,re
import deteval

def parse_args(arg_list):
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='dnn finetune')
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('-g', '--gpus', dest='gpus', help='GPU device id to use [0]',
            default='0')
    parser.add_argument('-j', '--jobfolder', required=False, default='', help='caffe model file')
    parser.add_argument('-i', '--intsv', required=True,   help='input tsv file for images, col_0:key, col_1:imgbase64')
    parser.add_argument('-t', '--iter', required=False, type=int,  help='snapshot iter')
    parser.add_argument('--mean', required=False, default='104,117,123', help='pixel mean value')
    args = parser.parse_args(arg_list)

    return args

class FileProgressingbar:
    fileobj = None
    pbar = None
    def __init__(self,fileobj):
        fileobj.seek(0,os.SEEK_END)
        flen = fileobj.tell()
        fileobj.seek(0,os.SEEK_SET)
        self.fileobj = fileobj
        widgets = ['Test: ', progressbar.AnimatedMarker(),' ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        self.pbar = progressbar.ProgressBar(widgets=widgets, maxval=flen).start()
    def update(self):
        self.pbar.update(self.fileobj.tell())

def vis_detections(im, prob, bboxes, labelmap, thresh=0.3, save_filename=None):
    """Visual debugging of detections."""
    im = im[:, :, (2, 1, 0)]
    plt.cla()
    fig = plt.imshow(im)

    for i, box in enumerate(bboxes):
        for j in range(prob.shape[1] - 1):
            if prob[i, j] < thresh:
                continue;
            score = prob[i, j]
            cls = j
            x,y,w,h = box
        
            im_h, im_w = im.shape[0:2]
            left  = (x-w/2.)
            right = (x+w/2.)
            top   = (y-h/2.)
            bot   = (y+h/2.)

            left = max(left, 0)
            right = min(right, im_w - 1)
            top = max(top, 0)
            bot = min(bot, im_h - 1)

            plt.gca().add_patch(
                plt.Rectangle((left, top),
                                right - left,
                                bot - top, fill=False,
                                edgecolor='g', linewidth=3)
                )
            plt.text(float(left), float(top - 10), '%s: %.3f'%(labelmap[cls], score), color='darkgreen', backgroundcolor='lightgray')
            #plt.title('{}  {:.3f}'.format(class_name, score))

    if save_filename is None:
        plt.show()
    else:
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(save_filename, bbox_inches='tight', pad_inches = 0)

def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    nparr = np.fromstring(jpgbytestring, np.uint8)
    try:
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR);
    except:
        return None;

def load_labelmap_list(filename):
    labelmap = []
    with open(filename) as fin:
        labelmap += [line.rstrip() for line in fin]
    return labelmap

def im_rescale(im, target_size):
    im_size_min = min(im.shape[0:2])
    im_size_max = max(im.shape[0:2])
    im_scale = float(target_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    return im

#def load_image_data(filename):
#    import struct
#    with open(filename, 'rb') as f:
#        w = struct.unpack('i', f.read(4))[0]
#        h = struct.unpack('i', f.read(4))[0]
#        data = np.fromfile(f, dtype=np.float32)
#        data = np.reshape(data, [3, h, w])
#        data = data[::-1, ...]
#    return data

def im_detect(net, im, pixel_mean, target_size=416):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= pixel_mean
    
    im_resized = im_rescale(im_orig, target_size)

    new_h, new_w = im_resized.shape[0:2]
    left = (target_size - new_w) / 2
    right = target_size - new_w - left
    top = (target_size - new_h) / 2
    bottom = target_size - new_h - top
    im_squared  = cv2.copyMakeBorder(im_resized, top=top, bottom=bottom, left=left, right=right, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    # change blob dim order from h.w.c to c.h.w
    channel_swap = (2, 0, 1)
    blob = im_squared.transpose(channel_swap)
    if pixel_mean[0] == 0:
        blob /= 255.

    # blob = load_image_data(r'detection-image.bin')      // for parity check

    net.blobs['data'].reshape(1, *blob.shape)
    net.blobs['data'].data[...]=blob.reshape(1, *blob.shape)
    net.blobs['im_info'].reshape(1,2)
    net.blobs['im_info'].data[...] = (im_orig.shape[0:2],)

    net.forward()

    bbox = net.blobs['bbox'].data[0]
    prob = net.blobs['prob'].data[0]

    return prob, bbox

def postfilter(im, scores, boxes, class_map, max_per_image=1000, thresh=0.005):
    class_num = scores.shape[1] - 1;        #the last one is obj_score * max_prob
    keep = np.where(scores[:, -1] > thresh)[0]
    scores = scores[keep, :]
    boxes = boxes[keep, :]

    image_scores = scores[:,-1]
    inds = np.argsort(-image_scores)
    scores = scores[inds, :]
    boxes = boxes[inds, :]

    # Limit to max_per_image detections
    if max_per_image > 0 and boxes.shape[0] > max_per_image:
        image_thresh = image_scores[max_per_image]
        keep = np.where(scores[:,-1] >= image_thresh)[0]
        scores = scores[0:max_per_image, :]
        boxes = boxes[0:max_per_image, :]

    det_results = [];
    for i, box in enumerate(boxes):
        crect = dict();

        x,y,w,h = box
        
        im_h, im_w = im.shape[0:2]
        left  = (x-w/2.);
        right = (x+w/2.);
        top   = (y-h/2.);
        bot   = (y+h/2.);

        left = max(left, 0)
        right = min(right, im_w - 1)
        top = max(top, 0)
        bot = min(bot, im_h - 1)

        crect['rect'] = map(float, [left,top,right,bot])
        cls = scores[i, 0:-1].argmax()
        crect['class'] = class_map[cls]
        crect['conf'] = float(scores[i, -1])
        det_results += [crect]
    
    return json.dumps(det_results)

def result2json(im, probs, boxes, class_map):
    class_num = probs.shape[1] - 1;        #the last one is obj_score * max_prob

    det_results = [];
    for i, box in enumerate(boxes):
        if probs[i, 0:-1].max() == 0:
            continue;
        for j in range(class_num):
            if probs[i,j] == 0:
                continue

            x,y,w,h = box
        
            im_h, im_w = im.shape[0:2]
            left  = (x-w/2.);
            right = (x+w/2.);
            top   = (y-h/2.);
            bot   = (y+h/2.);

            left = max(left, 0)
            right = min(right, im_w - 1)
            top = max(top, 0)
            bot = min(bot, im_h - 1)

            crect = dict();
            crect['rect'] = map(float, [left,top,right,bot])
            crect['class'] = class_map[j]
            crect['conf'] = float(probs[i, j])
            det_results += [crect]
    
    return json.dumps(det_results)

def detprocess(caffenet, caffemodel, pixel_mean, cmap, gpu, key_idx, img_idx, in_queue, out_queue):
    if gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(gpu)
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(caffenet, caffemodel, caffe.TEST)
    count = 0
    while True:
        cols = in_queue.get()
        if cols is None:
            print 'exiting: {0}'.format(in_queue.qsize())
            return 
        if len(cols)> 1:
            # Load the image
            im = img_from_base64(cols[img_idx])
            # Detect all object classes and regress object bounds
            scores, boxes = im_detect(net, im, pixel_mean)
            # vis_detections(im, scores, boxes, cmap, thresh=0.5)
            results = result2json(im, scores, boxes, cmap)
            out_queue.put(cols[key_idx] + "\t" + results+"\n")
            count = count + 1

def tsvdet(caffenet, caffemodel, intsv_file, key_idx,img_idx, pixel_mean, outtsv_file, **kwargs):
    if not caffemodel:
        caffemodel = op.splitext(caffenet)[0] + '.caffemodel'
    labelmapfile = 'labelmap.txt' if 'cmap' not in kwargs else kwargs['cmap']
    cmapfile = os.path.join(op.split(caffenet)[0], labelmapfile)
    if not os.path.isfile(cmapfile):
        cmapfile = os.path.join(os.path.dirname(intsv_file), 'labelmap.txt')
        assert os.path.isfile(cmapfile)
    if not os.path.isfile(caffemodel) :
        raise IOError(('{:s} not found.').format(caffemodel))
    if not os.path.isfile(caffenet) :
        raise IOError(('{:s} not found.').format(caffenet))
    cmap = load_labelmap_list(cmapfile)
    count = 0

    if 'gpus' in kwargs:
        gpus = [int(float(g)) for g in kwargs['gpus'].split(',')]
    else:
        gpus = [0]
    
    in_queue = mp.Queue(len(gpus)*2);  # thread/process safe
    out_queue = mp.Queue();
    worker_pool = [];
    for gpu in gpus:
        worker = mp.Process(target=detprocess, args=(caffenet, caffemodel,
            pixel_mean, cmap, gpu, key_idx, 
            img_idx, in_queue, out_queue));
        worker.daemon = True
        worker_pool.append(worker)
        worker.start()

    with open(intsv_file,"r") as tsv_in :
        bar = FileProgressingbar(tsv_in)
        for line in tsv_in:
            cols = [x.strip() for x in line.split("\t")]
            if len(cols) > img_idx:
                in_queue.put(cols)
                count = count + 1
                bar.update()

    for _ in worker_pool:
        in_queue.put(None)  # kill all workers

    outtsv_file_tmp = outtsv_file + '.tmp'
    with open(outtsv_file_tmp,"w") as tsv_out:
        for i in xrange(count):
            tsv_out.write(out_queue.get())

    for proc in worker_pool: #wait all process finished.
        proc.join()

    os.rename(outtsv_file_tmp, outtsv_file)
    caffe.print_perf(count)
    return count

if __name__ =="__main__":
    args = parse_args(sys.argv[1:])
    pixel_mean = [float(x) for x in args.mean.split(',')]
    #model_iter_150.caffemodel
    train_snapshots = glob.glob( op.join(args.jobfolder, "snapshot", "model_iter_*.caffemodel") )
    models =[(int(x.split('_')[-1].split('.')[0]), x) for x in train_snapshots]
    models = sorted(models, key=lambda x:x[0])
    proto = op.join(args.jobfolder,'test.prototxt')
    for (iter,model) in models:
        outfile = op.splitext(model)[0]+'.eval'
        if op.exists(outfile) : continue;
        if args.iter is not None and args.iter!=iter: continue;
        tsvdet(proto, model, args.intsv, 0, 2,  pixel_mean, outfile, gpus=args.gpus)

    iou=0.3
    perf = op.join(args.jobfolder,'test_%f.perf'%iou)
    with open(perf,'w') as logout:
        for (iter,model) in models:
            if args.iter is not None and args.iter!=iter: continue;
            outfile = op.splitext(model)[0]+'.eval'        
            # Get mAP report from outtsv over IoU threshold 0.5.
            report = deteval.eval(args.intsv, outfile, iou)
            # Get Precision/Recall over threhold 0.9.
            th, prec, rec = deteval.get_pr(report, 0.9)
            logout.write("%d\t%f\t%f\t%f\n"%(iter, report["map"], prec, rec))        
            logout.flush()
            print("%d\t%f\t%f\t%f"%(iter, report["map"], prec, rec))
            
    iou=0.5
    perf = op.join(args.jobfolder,'test_%f.perf'%iou)
    with open(perf,'w') as logout:
        for (iter,model) in models:
            if args.iter is not None and args.iter!=iter: continue;
            outfile = op.splitext(model)[0]+'.eval'
            # Get mAP report from outtsv over IoU threshold 0.5.
            report = deteval.eval(args.intsv, outfile, iou)
            # Get Precision/Recall over threhold 0.9.
            th, prec, rec = deteval.get_pr(report, 0.9)
            logout.write("%d\t%f\t%f\t%f\n"%(iter, report["map"], prec, rec))        
            logout.flush()
            print("%d\t%f\t%f\t%f"%(iter, report["map"], prec, rec))
