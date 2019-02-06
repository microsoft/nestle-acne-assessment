import numpy as np
import cv2
from skimage import io
import matplotlib.pyplot as plt
from os import listdir
from os.path import join, isfile, splitext
from scipy import misc
import sys
import dlib
import os
from PIL import Image
import urllib.request


width_ratio = 1.5
top_ratio = 1.5
gap_ratio = 0.1
down_ratio = 4.5
chin_width_ratio = 2.8
forehead_ratio = 0.3
verb = False

PREDICTOR_PATH = "../models/shape_predictor_68_face_landmarks.dat"
eye_cascade = cv2.CascadeClassifier('../models/haarcascade_eye.xml')

SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(im):
    rects = detector(im, 1)
    
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def read_imgURL(URL):
    with urllib.request.urlopen(URL) as url:
        with open('temp.jpg', 'wb') as f:
            f.write(url.read())

    img = Image.open('temp.jpg')
    img = np.array(img)
    return img

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im
    

def read_im_and_landmarks(fname):
    im = read_imgURL(fname)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)
    return im, s

def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def infer_chin_region(eye, width_ratio, down_ratio, left_or_right):
    region1 = [0] * 4
    if left_or_right == 'right': #assuming it is the absolute right chin
        region1[0] = int(max(0, int(eye[0] - 0.5 * eye[2]))) #chin region should go lefwards
        region1[2] = int(0.5 * eye[2])
    else: # assuming it is the absolute left chin
        region1[0] = int(eye[0] + eye[2]) # chin region should go rightwards
        region1[2] = int(0.5 * eye[2])
    region1[1] = int(eye[1] + eye[3])
    region1[3] = int(1.5 * eye[3])

    return region1
               
def detect_face_direction(gray, face, eye, down_ratio, chin_width_ratio):  
    region1 = [0] * 4 # assuming this is the left eye, forhead should go rightward
    region2 = [0] * 4 # assuming this is the right eye, forhead should go leftward
    print(eye[0])
    region1 = infer_chin_region(eye[0], chin_width_ratio, down_ratio, 'left') #region1 is from eye to right
    region2 = infer_chin_region(eye[0], chin_width_ratio, down_ratio, 'right') # region2 is from eye to left

    std1 = np.std(gray[region1[1]:(region1[1]+region1[3]), region1[0]:(region1[0]+region1[2])])
    std2 = np.std(gray[region2[1]:(region2[1]+region2[3]), region2[0]:(region2[0]+region2[2])])
    face_direction = ""

    if std1 > std2:  #eye right has higher variance than eye left
        face_direction = "right"
    else:
        face_direction = "left"
    return face_direction

def extract_cheek_region(face_x_min, face_x_max, face_y_max, eye_landmarks, left_or_right):
    if left_or_right == "Left":
        cheek_region_min_x = eye_landmarks[0,0]
        cheek_region_max_x = int(face_x_max - 0.05 * (face_x_max - min(eye_landmarks[:,0])))
    else:
        cheek_region_max_x = max(eye_landmarks[:,0])[0,0]
        #print (max(eye_landmarks[:,0])[0,0])
        #cheek_region_max_x = max(eye_landmarks[:, 0])
        cheek_region_min_x = int(face_x_min + 0.1 * (cheek_region_max_x - face_x_min))
    cheek_region_min_y = int(max(eye_landmarks[:,1]) + 0.2 * (max(eye_landmarks[:,1])  - min(eye_landmarks[:,1])))
    cheek_region_max_y = int(face_y_max - 0.1 * (face_y_max - max(eye_landmarks[:,1])))
    return [cheek_region_min_x, cheek_region_min_y, cheek_region_max_x, cheek_region_max_y]

def extract_patches(imagefile, dimension_dict, face_loc_dict, image_dim, croppedFaces_Dir):
    
    imageName = splitext(os.path.basename(imagefile))[0]
    
    try:
        img, landmarks = read_im_and_landmarks(imagefile)
        face_detected = True
    except:
        img = read_imgURL(imagefile)
        face_detected = False
     
    img_height, img_width = img.shape[0:2]
    image_dim = [img_height, img_width]
    min_dim = min(img_height, img_width)
    min_face_size = min(min_dim * 0.2, min_dim * 0.2)
    min_eye = min_face_size * 0.2
    min_eye_area = min_eye ** 2
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if face_detected:
        mask = get_face_mask(img, landmarks)
        face_x_min = int(max(0, np.asarray(min(landmarks[:,0])).flatten()[0]))
        face_x_max = int(min(img_width, np.asarray(max(landmarks[:,0])).flatten()[0]))
        face_y_min = int(max(0, np.asarray(min(landmarks[:,1])).flatten()[0]))
        face_y_max = int(min(img_height, np.asarray(max(landmarks[:,1])).flatten()[0]))
        face_loc_dict['face_loc'] = [face_x_min, face_x_max, face_y_min, face_y_max]
        face_height = face_y_max - face_y_min
        forehead_height = int(face_height * forehead_ratio)
        new_face_y_min = max(0, face_y_min - forehead_height)
        right_brow_landmarks = landmarks[RIGHT_BROW_POINTS,:]
        left_brow_landmarks = landmarks[LEFT_BROW_POINTS,:]
        right_eye_landmarks = landmarks[RIGHT_EYE_POINTS,:]
        left_eye_landmarks = landmarks[LEFT_EYE_POINTS,:]
        mouse_landmarks = landmarks[MOUTH_POINTS,:]
        ########################
        # Get the forehead patch
        ########################
        [right_brow_min_x, left_brow_max_x] = \
            [max(0, np.min(np.array(right_brow_landmarks[:,0]))), min(img_width, np.max(np.array(left_brow_landmarks[:,0])))]
        brow_min_y = min(np.min(np.array(right_brow_landmarks[:,1])),np.min(np.array(left_brow_landmarks[:,1])))
        forehead_x_min = right_brow_min_x
        forehead_x_max = left_brow_max_x
        forehead_y_min = max(0, brow_min_y - forehead_height)
        forehead_y_max = min(brow_min_y, forehead_y_min + forehead_height)
        forehead_region = img[forehead_y_min:forehead_y_max, forehead_x_min:forehead_x_max, :]
        #print ('forehead dim (x_min, x_max, y_min, y_max): %i,%i, %i, %i' % (forehead_x_min,  forehead_x_max,  forehead_y_min,  forehead_y_max))
        key_name = 'landmark_fh'
        dimension_dict[key_name] = [forehead_x_min, forehead_x_max, forehead_y_min, forehead_y_max]
        forehead_file_name = join(croppedFaces_Dir, key_name +".jpg")
        #forehead_region = cv2.cvtColor(forehead_region, cv2.COLOR_BGR2RGB)
        misc.imsave(forehead_file_name, forehead_region)
        
        chin_x_min = np.max(np.array(right_eye_landmarks[:,0]))
        chin_x_max = np.min(np.array(left_eye_landmarks[:,0]))
        chin_y_min = np.max(np.array(mouse_landmarks[:,1]))
        chin_y_max = face_y_max
        chin_region = img[chin_y_min:chin_y_max, chin_x_min:chin_x_max, :]
        #print ('chin dim (x_min, x_max, y_min, y_max): %i,%i, %i, %i' % (chin_x_min, chin_x_max, chin_y_min, chin_y_max))
        key_name = 'landmark_chin'
        dimension_dict[key_name] = [chin_x_min, chin_x_max, chin_y_min, chin_y_max]
        chin_file_name = join(croppedFaces_Dir, key_name +".jpg")
        #chin_region = cv2.cvtColor(chin_region, cv2.COLOR_BGR2RGB)
        misc.imsave(chin_file_name, chin_region)
    
        ##########################
        # Get the cheeks patch
        ##########################
        # Decide whether it is a side view or not
        left_eye_width = np.max(np.array(left_eye_landmarks[:,0])) - np.min(np.array(left_eye_landmarks[:,0]))
        right_eye_width = np.max(np.array(right_eye_landmarks[:,0])) - np.min(np.array(right_eye_landmarks[:,0]))
        right_face = True
        left_face = True
        if float(right_eye_width) / float(left_eye_width) >= 1.15: # right eye is bigger than left eye, showing the right face
            left_face = False
        elif float(left_eye_width) / float(right_eye_width) >= 1.15: # left eye is bigger than right eye, showing the left face
            right_face = False
        
        if right_face:
            right_cheek_region = extract_cheek_region(face_x_min, face_x_max, face_y_max, right_eye_landmarks, "Right")
            cheek_region = img[right_cheek_region[1]:right_cheek_region[3], right_cheek_region[0]:right_cheek_region[2], :]
            #print ('right cheek dim (x_min, x_max, y_min, y_max): %i,%i, %i, %i' % (right_cheek_region[0], right_cheek_region[2], right_cheek_region[1], right_cheek_region[3]))
            key_name = 'landmark_rc'
            dimension_dict[key_name] = [right_cheek_region[0], right_cheek_region[2], right_cheek_region[1], right_cheek_region[3]]
            cheek_file_name = join(croppedFaces_Dir, key_name +".jpg")
            #cheek_region = cv2.cvtColor(cheek_region, cv2.COLOR_BGR2RGB)
            misc.imsave(cheek_file_name, cheek_region)
        if left_face:
            left_cheek_region = extract_cheek_region(face_x_min, face_x_max, face_y_max, left_eye_landmarks, "Left")
            cheek_region = img[left_cheek_region[1]:left_cheek_region[3], left_cheek_region[0]:left_cheek_region[2], :]
            #print ('left cheek dim (x_min, x_max, y_min, y_max): %i,%i, %i, %i' % (left_cheek_region[0], left_cheek_region[2], left_cheek_region[1], left_cheek_region[3]))
            key_name = 'landmark_lc'
            dimension_dict[key_name] = [left_cheek_region[0], left_cheek_region[2], left_cheek_region[1], left_cheek_region[3]]
            cheek_file_name = join(croppedFaces_Dir, key_name +".jpg")
            #cheek_region = cv2.cvtColor(cheek_region, cv2.COLOR_BGR2RGB)
            misc.imsave(cheek_file_name, cheek_region)
    
                
    if not face_detected:
        print("Face not detected by landmarks model...")
        # Use the OneEye model to detect one eye, and infer the face region based on the eye location
        eye_detected = False
        roi_gray = gray
        roi_color = img
        roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        max_area = 0
        eye_count = 0
        max_index = 0
        
        for (ex,ey,ew,eh) in eyes: # there might be multiple eyes detected. Choose the biggest one
            if ew*eh >= max_area and ex >= img_width * 0.1 and ex <= img_width * 0.9:
                max_area = ew*eh
                max_index = eye_count
            eye_count += 1
        if max_area >= min_eye_area:
            eye_detected = True
            (ex, ey, ew, eh) = eyes[max_index]
            if float(ew) / float(img_width) > 0.15 or float(eh) / float(img_height) > 0.15: # detected eye too large
                # resize the detected eye
                center_x = ex + ew/2
                center_y = ey + eh/2
                resized_w = min(img_width * 0.15, img_height * 0.15)
                ex = int(center_x - resized_w/2)
                ey = int(center_y - resized_w/2)
                ew = int(resized_w)
                eh = int(resized_w)
                eyes1 = np.array([ex, ey, resized_w, resized_w]).reshape((1,4))
            else:
                eyes1 = np.array(eyes[max_index]).reshape((1,4))
            face1 = np.array(())
            face_direction = detect_face_direction(gray, face1, eyes1, down_ratio, chin_width_ratio)
            if face_direction == "left":
                print("Left eye detected")
                face_min_x = eyes1[0, 0]
                face_max_x = min(img_width, int(eyes1[0,0] + (chin_width_ratio + 0.5) * eyes1[0, 2]))
                forehead_max_x = min(img_width, int(eyes1[0,0] + width_ratio * eyes1[0, 2]))
                forehead_min_x = face_min_x
                cheek_min_x = int(eyes1[0, 0] + 0.5 * eyes1[0,2])
                cheek_max_x = face_max_x
            else:
                print("Right eye detected")
                face_min_x = max(0, int(eyes1[0, 0] - chin_width_ratio * eyes1[0, 2]))
                face_max_x = eyes1[0, 0] + eyes1[0, 2]
                forehead_min_x = max(0, int(eyes1[0, 0] - width_ratio * eyes1[0, 2]))
                forehead_max_x = min(img_width, int(eyes1[0, 0] + width_ratio * eyes1[0, 2]))   
                cheek_max_x = int(eyes1[0,0] + 0.5*eyes1[0,2])
                cheek_min_x = face_min_x
            forehead_min_y = max(0, int(eyes1[0, 1] - top_ratio * eyes1[0,3]))
            forehead_max_y = max(0, int(eyes1[0, 1] - 0.5 * eyes1[0, 3]))
            forehead_ok = False
            # Get the forehead region
            if forehead_max_y - forehead_min_y >= 0.7 * eyes1[0, 3]:
                forehead_ok = True
                forehead_region = img[forehead_min_y:forehead_max_y, forehead_min_x: forehead_max_x, :]
                #print ('forehead dim (x_min, x_max, y_min, y_max): %i,%i, %i, %i' % (forehead_min_x,  forehead_max_x,  forehead_min_y,  forehead_max_y))
                key_name = 'oneeye_fh'
                dimension_dict[key_name] = [forehead_min_x,  forehead_max_x,  forehead_min_y,  forehead_max_y]
                forehead_file_name = join(croppedFaces_Dir, key_name +".jpg")
                misc.imsave(forehead_file_name, forehead_region)
            # Get the cheek region
            cheek_min_y = int(eyes1[0, 1] + eyes1[0, 3])
            cheek_max_y = min(img_height, int(eyes1[0, 1] + down_ratio * eyes1[0, 3]))
            cheek_region = img[cheek_min_y: cheek_max_y, cheek_min_x: cheek_max_x, :]
            #print ('cheek dim (x_min, x_max, y_min, y_max): %i,%i, %i, %i' % (cheek_min_x, cheek_max_x, cheek_min_y, cheek_max_y))
            key_name = 'oneeye_cheek'
            dimension_dict[key_name] = [cheek_min_x, cheek_max_x, cheek_min_y, cheek_max_y]
            face_loc_dict['face_loc'] = [face_min_x, face_max_x, forehead_min_y, cheek_max_y]
            #cheek_region = cv2.cvtColor(cheek_region, cv2.COLOR_BGR2RGB)
            if face_direction == "left":
                cheek_file_name = join(croppedFaces_Dir, key_name +".jpg")
            elif face_direction == "right":
                cheek_file_name = join(croppedFaces_Dir, key_name +".jpg")
            else:
                cheek_file_name = join(croppedFaces_Dir, key_name +".jpg")
            misc.imsave(cheek_file_name, cheek_region)
    
    
    if (not face_detected) and (not eye_detected):
        print("No chin or forehead detected, output the original file %s.jpg"%imageName)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        outfile = join(croppedFaces_Dir, imageName+".jpg")
        misc.imsave(outfile, img)
    
    return dimension_dict, face_loc_dict, image_dim
