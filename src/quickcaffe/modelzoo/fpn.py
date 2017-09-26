from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
from layerfactory import *
#from resnet import ResNet
from darknet import DarkNet
class FeaturePyramidNetwork(object):
    def add_body(self, n, stages,featuredim, lr=1, deploy=True):
        retlist = [];
        for i in range(3):
            sname = stages[-i-1][0]
            bottom = n[sname];
            if i==0:
                n[sname+'_feature'] = conv(bottom, nout=featuredim, ks=1, stride=1, pad=0, lr=lr, deploy=deploy)
            else:
                n[sname+'_fpn1x1'] = conv(bottom, nout=featuredim, ks=1, stride=1, pad=0, lr=lr, deploy=deploy)
                n[sname+'_prev'] = deconv(n[prev_name],nout=featuredim, ks=3, stride=2, pad=1, lr = lr, deploy=deploy)
                n[sname+'_feature'] = L.Eltwise(n[sname+'_fpn1x1'], n[sname+'_prev'])
            prev_name =sname+'_feature'
            retlist += [(prev_name, stages[-i-1][1],stages[-i-1][2])]
        return retlist
        
def to_proto_str(n, test_str='', data_str='', deploy=False):
    layers = str(n.to_proto()).split('layer {')[1:]
    layers = ['layer {' + x for x in layers]

    return ''.join(layers)            
if __name__ == "__main__":
    fpn = FeaturePyramidNetwork()
    # add body and extra
    #model = ResNet()
    model = DarkNet()
    fpn = FeaturePyramidNetwork()
    n = caffe.NetSpec()
    n.data = caffe.layers.Layer() 
    deploy = True
    model.add_body(n, lr=1, depth=19, deploy=deploy)
    stages = model.get_stages()
    feature_stages = fpn.add_body(n, stages, 256,lr=1, deploy=deploy)
    print(feature_stages)
    proto_str = to_proto_str(n,  deploy=deploy)
    with open('test.prototxt','w') as protof:
        protof.write(proto_str)

