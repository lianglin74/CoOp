import sys, os
import caffe
import h5py

def readh (hdf5_file):
    arrays = dict();
    with h5py.File(hdf5_file,'r') as f:
        for key in f:
            arrays[key.replace('\\','/')] = np.array(f[key])
    return arrays;

class HDF5DataLayer(caffe.Layer):
  def setup(self,bottom,top):
    # read parameters from `self.param_str`
    params = eval(self.param_str)
    feature_path = params['datafile']
    self.batch_size = params['batch_size']
    self.iter=0;
    
    npzfile = readh(feature_path)
    for key,value in npzfile.items():
        if key=='label':
            self.Y = value.reshape(-1,self.batch_size, value.shape[1])
        else:
            self.X = value.reshape(-1,self.batch_size, value.shape[1], value.shape[2], value.shape[3])
            
  def reshape(self,bottom,top):
    # no "bottom"s for input layer
    if len(bottom)>0:
      raise Exception('cannot have bottoms for input layer')
    # make sure you have the right number of "top"s
    if len(top)!= 2:
       raise "Only two top layers, data and label"
    top[0].reshape( self.batch_size, self.X.shape[2],self.X.shape[3],self.X.shape[4] ) # reshape the outputs to the proper sizes
    top[1].reshape( self.batch_size, self.X.shape[2] ) # reshape the outputs to the proper sizes

  def forward(self,bottom,top): 
    # do your magic here... feed **one** batch to `top`
    top[0].data[...] = self.X[self.iter,:,:,:,:]
    top[1].data[...] = self.Y[self.iter,:,:]
    self.iter = (self.iter+1)%(self.X.shape[0])
  def backward(self, top, propagate_down, bottom):
    # no back-prop for input layers
    pass