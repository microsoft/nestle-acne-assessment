
import getPatches
from regressionModel import extract_features, predict_label
import os
import shutil

def extract_patches(imageURL):
    patch_path = 'patches'
    dimension_dict = dict()
    face_dict = dict()
    image_dim = []
    try:
        dim, face, img = getPatches.extract_patches(imageURL, dimension_dict,face_dict, image_dim, patch_path)
        print ("extract patches pass")
    except:
        print ('cannot extract patches from the image')
    return dim, face, img

def score_patch(patch_path):
    patch_score = dict()
    for file in next(os.walk(patch_path))[2]:
        file_path = os.path.join(patch_path, file)    
        score_features = extract_features (file_path)[0].flatten()# extract features from CNTK pretrained model      
        pred_score_label = predict_label(score_features) # score the extracted features using trained regression model
        patch_score[file.split('.')[0]] = float("{0:.2f}".format(pred_score_label[0]))
    return patch_score


def infer_label(patch_score, label_mapping):
    max_score_name, max_score_value = max(patch_score.items(), key=lambda x:x[1])
    pred_label = label_mapping[round(max_score_value)-1]
    return pred_label

def del_cache(patch_folder):
    shutil.rmtree(patch_folder)
    return