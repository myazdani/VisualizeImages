import sys
sys.path.insert(0, "../SkiCaffe/")
from skicaffe import SkiCaffe

caffe_root = '/usr/local/src/caffe/caffe-master/'
DLmodel = SkiCaffe('/usr/local/src/caffe/caffe-master/')
model_prototxt = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_trained = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

DLmodel.fit(model_prototxt_path = model_prototxt, model_trained_path = model_trained)

for key in DLmodel.layer_dict.keys():
  if key == 'prob' or key == 'data':
    continue
  image_features = DLmodel.transform(image_paths = image_paths, layer_name = key, return_type = 'pandasDF')
  image_features.to_csv("../features/BVLCref_" + key + ".csv")