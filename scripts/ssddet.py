#!python2
import os.path as op
import _init_paths
import numpy as np
import caffe, os, sys, cv2
import argparse
import numpy as np
import base64
import progressbar
import json
import matplotlib.pyplot as plt
import multiprocessing as mp

def parse_args(arg_list):
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SSD detection')
    parser.add_argument('--gpus', help='GPU device id to use [0]', type=int, nargs='+')
    parser.add_argument('--net', dest='net', help='Network to use')
    parser.add_argument('--model', required=False, default='', help='caffe model file')
    parser.add_argument('--intsv', required=True,   help='input tsv file for images, col_0:key, col_1:imgbase64')
    parser.add_argument('--colkey', required=False, type=int, default=0,  help='key col index')
    parser.add_argument('--colimg', required=False, type=int, default=1,  help='imgdata col index')
    parser.add_argument('--outtsv', required=False, default="",  help='output tsv file with roi info')
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

def im_rescale_to_square(im, target_size):
    im = cv2.resize(im, (target_size, target_size),
                    interpolation=cv2.INTER_LINEAR)
    return im

def im_detect(net, im, pixel_mean, target_size=300, **kwargs):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= pixel_mean

    im_squared = im_rescale_to_square(im_orig, target_size)

    # change blob dim order from h.w.c to c.h.w
    channel_swap = (2, 0, 1)
    blob = im_squared.transpose(channel_swap)
    if pixel_mean[0] == 0:
        blob /= 255.

    net.blobs['data'].reshape(1, *blob.shape)
    net.blobs['data'].data[...]=blob.reshape(1, *blob.shape)

    net.forward()

    # detection_out from ssd is of shape (1,1,N,7), where N is the number of bboxes
    detect_out = net.blobs['detection_out'].data[0, 0]

    prob = []
    bbox = []

    for i in range(detect_out.shape[0]):
        out = detect_out[i]
        assert out[0] == 0  # assume single image prediction
        prob.append(out[1:3])   # cls_id and pred confidence
        bbox.append(out[3:])  # xyxy in relative scale [0,1] without clip

    return prob, bbox

def result2json(im, probs, boxes, class_map):
    det_results = []
    for i, box in enumerate(boxes):
        cls_id, conf = probs[i]
        if cls_id == 0 or conf == 0:    # if background or conf 0, skip
            continue

        im_h, im_w = im.shape[0:2]
        left, top, right, bot = box
        # scale
        left *= im_w
        right *= im_w
        top *= im_h
        bot *= im_h
        # clip
        left = max(left, 0)
        right = min(right, im_w - 1)
        top = max(top, 0)
        bot = min(bot, im_h - 1)
        # # output in ssd is 1-based. Decrease 1 for quickdetection format.
        # left = max(left-1, 0)
        # right = min(right-1, im_w-1)
        # top = max(top-1, 0)
        # bot = min(bot-1, im_h-1)

        crect = dict()
        crect['rect'] = map(float, [left,top,right,bot])
        crect['class'] = class_map[int(cls_id)]
        crect['conf'] = float(conf)
        det_results += [crect]
    
    return json.dumps(det_results)

def detprocess(caffenet, caffemodel, pixel_mean, cmap, gpu, key_idx, img_idx,
        in_queue, out_queue, **kwargs):
    if gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(gpu)
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(str(caffenet), str(caffemodel), caffe.TEST)
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
            scores, boxes = im_detect(net, im, pixel_mean, **kwargs)
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

    gpus = kwargs.get('gpus', [0])
    
    in_queue = mp.Queue(len(gpus)*2);  # thread/process safe
    out_queue = mp.Queue()
    worker_pool = []
    for gpu in gpus:
        worker = mp.Process(target=detprocess, args=(caffenet, caffemodel,
            pixel_mean, cmap, gpu, key_idx, 
            img_idx, in_queue, out_queue), kwargs=kwargs);
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
    outtsv_file = args.outtsv if args.outtsv!="" else os.path.splitext(args.intsv)[0]+".eval"

    pixel_mean = [float(x) for x in args.mean.split(',')]

    tsvdet(args.net, args.model, args.intsv, args.colkey, args.colimg,
            pixel_mean, outtsv_file, gpus=args.gpus)
