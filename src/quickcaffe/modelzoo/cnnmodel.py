from layerfactory import *

class CNNModel(object):
    def __init__(self, add_last_pooling_layer=True, rcnn_mode = False):
        # if rcnn_mode is true, add_body(...) will make the last stage a higher resolution of 14x14
        # to ensure the feature map size is divided by 16, not by 32, assuming the input_size being 224.
        self.add_last_pooling_layer = add_last_pooling_layer
        self.rcnn_mode = rcnn_mode

    def crop_size(self):
        return 224

    def roi_size(self):
        return 6

    def add_body(self, netspec, depth=-1, lr=1, deploy=True):
        pass
    
    def add_extra(self, netspec, num_classes, lr, lr_lastlayer, deploy=True):
        pass

    def add_loss(self, netspec, accuracy_top5):
        n = netspec
        last = last_layer(n)
        n.loss = L.SoftmaxWithLoss(last, n.label)
        n.accuracy = L.Accuracy(last, n.label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
        if accuracy_top5:
            n.accuracy_top_5 = L.Accuracy(last, n.label, include=dict(phase=getattr(caffe_pb2, 'TEST')), accuracy_param=dict(top_k=5))

    def add_prediction(self, netspec):
        netspec.prob = L.Softmax(last_layer(netspec))

    def add_loss_or_prediction(self, netspec, deploy, accuracy_top5):
        if deploy:
            self.add_prediction(netspec)
        else:
            self.add_loss(netspec, accuracy_top5)

    def add_body_for_feature(self, netspec, depth=-1, lr=1, deploy=True):
        pass

    def add_body_for_roi(self, netspec, bottom, lr=1, deploy=True):
        pass
