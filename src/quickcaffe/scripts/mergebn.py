import numpy as np
import sys
import os
import os.path as osp
import google.protobuf as pb
from argparse import ArgumentParser
import caffe

def process_prototxt(proto_old, proto_new):
    #load model and weights
    with open(proto_old) as f:
        model = caffe.proto.caffe_pb2.NetParameter()
        pb.text_format.Parse(f.read(), model)
    
    # Get the BN layers to be absorbed
    to_be_absorbed = []
    for i, layer in enumerate(model.layer):
        if layer.type != 'BatchNorm': continue
        bottom = layer.bottom[0]
        top = layer.top[0]
        convlayer = model.layer[i-1];
        if len(convlayer.param)==1:   # training prototxt
            convlayer.param.add(lr_mult=2,decay_mult=0)  #add learning rate and decay param for bias term
        convlayer.convolution_param.bias_term = True
        scale_layer = model.layer[i+1]
        if bottom not in convlayer.top or convlayer.type not in ['Convolution', 'InnerProduct']:
            continue
        if top not in scale_layer.bottom or scale_layer.type not in ['Scale']:
            continue;
        scale_top = scale_layer.top[0];
        to_be_absorbed += [layer.name, scale_layer.name]
        # Rename the top blobs
        for j in xrange(i + 2, len(model.layer)):
            top_layer = model.layer[j]
            if scale_top in top_layer.bottom:
                names = list(top_layer.bottom)
                names[names.index(scale_top)] = bottom
                del(top_layer.bottom[:])
                top_layer.bottom.extend(names)
            if scale_top in top_layer.top:
                names = list(top_layer.top)
                names[names.index(scale_top)] = bottom
                del(top_layer.top[:])
                top_layer.top.extend(names)

    # Save the prototxt
    output_model_layers = [layer for layer in model.layer
                           if layer.name not in to_be_absorbed]
    output_model = caffe.proto.caffe_pb2.NetParameter()
    output_model.CopyFrom(model)
    del(output_model.layer[:])
    output_model.layer.extend(output_model_layers)
    with open(proto_new, 'w') as f:
        f.write(pb.text_format.MessageToString(output_model))
    return model, to_be_absorbed    
    
def process_weights(weights_old, weights_new, to_be_absorbed, proto_old,proto_new, model):
    # Absorb the BN parameters
    weights = caffe.Net(proto_old, weights_old , caffe.TEST)
    # Save the caffemodel
    output_weights = caffe.Net(proto_new, caffe.TEST)
    bias_dict = dict();
    for i, layer in enumerate(model.layer):
        if layer.type!='BatchNorm' or layer.name not in to_be_absorbed: 
            continue
        mean, var, aggcnt = [p.data.ravel()  for p in weights.params[layer.name]]
        scale_factor = 1.0/aggcnt[0] if aggcnt[0]>0 else 0;
        mean *= scale_factor;
        var *= scale_factor;
        scale,bias = [p.data.ravel()  for p in weights.params[model.layer[i+1].name]]
        eps = 1e-5
        invstd = 1./np.sqrt( var + eps )
        invstd = invstd*scale

        for j in xrange(i - 1, -1, -1):
            bottom_layer = model.layer[j]
            if layer.bottom[0] in bottom_layer.top:
                W = weights.params[bottom_layer.name][0]
                num = W.data.shape[0]
                if bottom_layer.type == 'Convolution':
                    W.data[...] = (W.data * invstd.reshape(num,1, 1,1))
                    bias_dict[bottom_layer.name] = (- mean) * invstd + bias
    for name in output_weights.params:
        if name not in bias_dict:
            for i in xrange(len(output_weights.params[name])):
                output_weights.params[name][i].data[...] = weights.params[name][i].data.copy()
        else:
             output_weights.params[name][0].data[...] = weights.params[name][0].data.copy()
             output_weights.params[name][1].data[...] = bias_dict[name]
    output_weights.save(weights_new)


if __name__ == '__main__':
    parser = ArgumentParser(
            description="Generate Batch Normalized model for inference")
    parser.add_argument('-m', '--model', help="The net definition prototxt", required=True)
    parser.add_argument('-w', '--weights', help="The weights caffemodel", default=None, required=False)
    parser.add_argument('-om', '--output_model', help = 'New net definition prototxt')
    parser.add_argument('-ow', '--output_weights', help="New weights caffemodel")
    args = parser.parse_args()
    proto_old = args.model;
    proto_new = args.output_model or osp.splitext(proto_old)[0] + '_inference.prototxt'
    model, to_be_absorbed = process_prototxt(proto_old, proto_new);
    if args.weights is not None:
        weights_old = args.weights;
        weights_new = args.output_weights or osp.splitext(weights_old)[0] + '_inference.caffemodel'
        process_weights(weights_old,weights_new,to_be_absorbed, proto_old,proto_new,model)