
import numpy as np
import pandas as pd
import cntk as C
from PIL import Image
import pickle
from cntk import load_model, combine
import cntk.io.transforms as xforms
from cntk.logging import graph
from cntk.logging.graph import get_node_outputs


pretrained_model = '../models/ResNet152_ImageNet_Caffe.model'
pretrained_node_name = 'pool5'
regression_model = '../models/cntk_regression.dat'
image_width = 224
image_height = 224

# load CNTK pretrained model
#model_file  = os.path.join(pretrained_model_path, pretrained_model_name)
loaded_model  = load_model(pretrained_model) # a full path is required
node_in_graph = loaded_model.find_by_name(pretrained_node_name)
output_nodes  = combine([node_in_graph.owner])

# load the stored regression model
read_model = pd.read_pickle(regression_model)
regression_model = read_model['model'][0]
train_regression = pickle.loads(regression_model)

def extract_features(image_path):   
    img = Image.open(image_path)       
    resized = img.resize((image_width, image_height), Image.ANTIALIAS)  
    
    bgr_image = np.asarray(resized, dtype=np.float32)[..., [2, 1, 0]]    
    hwc_format = np.ascontiguousarray(np.rollaxis(bgr_image, 2)) 
    
    arguments = {loaded_model.arguments[0]: [hwc_format]}    
    output = output_nodes.eval(arguments)   
    return output

def predict_label(features):
    return train_regression.predict(features.reshape(1,-1))