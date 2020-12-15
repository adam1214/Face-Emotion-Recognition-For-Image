#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 17:33:00 2020

@author: peterwu and Chun-Yu Chen

"""
# import sys
from statistics import mode
import time

import glob
import csv
import codecs
import shutil
import os
import cv2
import numpy as np
import torch

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from models import googlenet as model
from torchvision.transforms import transforms
from PIL import Image
import torch.nn.functional as F

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#Detecting face emotion on input image

def fun(in_path, out_image_path,
        out_info_path, in_finished_path,
        model_path, image_resolution):
    """
     >>>  fun(/fer_input, /fer_output, /fer_result, /fer_finished, /fer_model, image_resolution)
    .jpg files in the fer_intput folder will move to fer_finished folder.
    Processed .jpg files will be saved in fer_output folder.
    .csv files will be saved in fer_result folder.
    only process the image that its resolution is 720p and above(image_resolution = 720, can be adjusted)
    """
    global model, F
    detect_emo = True

    #save config
    save_image = True
    save_info = True
    show_image = False

    #config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    # config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    #%%
    # parameters for loading data and images
    detection_model_path = model_path +  '/haarcascade_frontalface_default.xml'
    if detect_emo:
        emotion_model_path = model_path + '/googlenet__googlenetwei__2020Aug29_16.21'
        emotion_labels = get_labels('fer2013')
        print(emotion_labels)
        emotion_offsets = (20, 40)

        # loading models
        model = getattr(model, 'googlenet')
        model = model(in_channels=3, num_classes=7)
        #print(torch.cuda.is_available())
        #print(torch.cuda.device_count())
        state = torch.load(emotion_model_path, map_location='cpu')
        model.load_state_dict(state['net'])

        #model.cuda()
        model.eval()

        # getting input model shapes for inference
        emotion_target_size = (224,224)
        # starting lists for calculating modes
        emotion_window = []

    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)

    # loading models
    face_detection = load_detection_model(detection_model_path)

    info_name = ['file name', 'face_x', 'face_y', 'face_w', 'face_h', 'emotion', 'angry_prob', 'disgust_prob', 'fear_prob', 'happy_prob', 'sad_prob', 'surprise_prob', 'neutral_prob']

    input_image_root = in_path
    output_image_root = out_image_path
    output_info_root = out_info_path

    #covert png to jpg
    for image_path in glob.glob(input_image_root+'/**/*.png', recursive=True):
        img_path_no_PNG = image_path.split('.png')[0]
        image_name = image_path.split('/')[-1].split('.png')[0]
        no_root_path = image_path[len(input_image_root):].replace(image_path.split('/')[-1], '')
        im = Image.open(image_path)
        rgb_im = im.convert('RGB')
        rgb_im.save(img_path_no_PNG + '.jpg')

        #delete PNG file
        os.remove(image_path)

    for image_path in glob.glob(input_image_root+'/**/*.jpg', recursive=True):
        print(image_path)
        no_root_path = image_path[len(input_image_root):].replace(image_path.split('/')[-1], '')
        spec_csv_name = ('/' + (image_path[len("feri_input/"):].replace(image_path.split('/')[-1], ''))).split('/')[-2]
        if spec_csv_name =="":
            spec_csv_name = "result.csv"
        else:
            spec_csv_name = spec_csv_name + '.csv'
        image_capture = cv2.imread(image_path)
        image_cap_ori = image_capture
        image_name = image_path.split('/')[-1].split('.jpg')[0]
        ori_image_name = image_path.split('/')[-1]
        size = (round(image_capture.shape[0]), round(image_capture.shape[1])) # float
        ori_size = size
        reduce_resolution = 0
        first_construct_csv = False
        scaling_factor_x = 1
        scaling_factor_y = 1

        if image_resolution == "720p" and size[0] > 720 and size[1] > 1280:
            #need to reduce resolution to 720p
            reduce_resolution = 1
            out_path = input_image_root + no_root_path+'resize_to_720p_'+image_path.split('/')[-1]
            image_capture = cv2.resize(image_capture, (1280, 720), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(out_path, image_capture)

            scaling_factor_y = size[0]/720
            scaling_factor_x = size[1]/1280

            #original resolution image move to fer_finished dir 
            src = image_path
            dst = in_finished_path + no_root_path + image_name + ".jpg"
            os.makedirs(os.path.dirname(in_finished_path + no_root_path), exist_ok=True)
            shutil.move(src, dst)

            #capture ori resolution image to draw bounding box
            image_cap_ori = cv2.imread(dst)

            #capture reducing resolution image to construct csv file
            image_path = out_path
            image_capture = cv2.imread(image_path)
            image_name = image_path.split('/')[-1].split('.jpg')[0]
            size = (round(image_capture.shape[0]), round(image_capture.shape[1])) # float

        if size[0] == 720 and size[1] == 1280:
            if save_image:
                os.makedirs(os.path.dirname(output_image_root + no_root_path), exist_ok=True)
                out_path = output_image_root+no_root_path+ori_image_name
            if save_info:
                os.makedirs(os.path.dirname(output_info_root + no_root_path), exist_ok=True)
                if os.path.isfile(output_info_root+no_root_path+spec_csv_name) == False:
                    first_construct_csv = True
                csv_info = codecs.open(
                    output_info_root+no_root_path+spec_csv_name, 'a', encoding="utf_8_sig"
                )
                csv_writer = csv.writer(csv_info)
                if first_construct_csv == True:
                    csv_writer.writerow(info_name)

            st_time = time.time()
            
            gray_image = cv2.cvtColor(image_capture, cv2.COLOR_BGR2GRAY)

            faces = detect_faces(face_detection, gray_image)
            if not isinstance(faces, tuple):
                faces = faces[faces[:,0].argsort()]
                faces = faces[faces[:,1].argsort()]
                faces = faces[faces[:,2].argsort()]
                faces = faces[faces[:,3].argsort()]

            for face_coordinates in faces:
                x_1, x_2, y_1, y_2 = apply_offsets(face_coordinates, emotion_offsets)

                if detect_emo:
                    gray_face = gray_image[y_1:y_2, x_1:x_2]
                    try:
                        gray_face = cv2.resize(gray_face, (emotion_target_size))
                    except:
                        continue

                    gray_face = np.dstack([gray_face] * 3)
                    
                    gray_face = transforms.Compose([ transforms.ToPILImage(),transforms.ToTensor(),])(np.uint8(gray_face))
                    
                    gray_face = torch.stack([gray_face], 0) 
                    #gray_face = gray_face.cuda(non_blocking=True)
                    outputs = model(gray_face).cpu()
                    outputs = F.softmax(outputs, 1)

                    emotion_prediction = torch.sum(outputs, 0).cpu().detach().numpy()  # outputs.shape [tta_size, 7]
                    emotion_probability = np.max(emotion_prediction)
                    emotion_label_arg = np.argmax(emotion_prediction)
                    emotion_text = emotion_labels[emotion_label_arg]
                    emotion_window.append(emotion_text)

                    if len(emotion_window) > frame_window:
                        emotion_window.pop(0)
                    '''
                    try:
                        emotion_mode = mode(emotion_window)
                    except:
                        continue
                    '''
                    x = int(float(face_coordinates[0]*scaling_factor_x))
                    y = int(float(face_coordinates[1]*scaling_factor_y))
                    w = int(float(face_coordinates[2]*scaling_factor_x))
                    h = int(float(face_coordinates[3]*scaling_factor_y))
                    if emotion_text == 'angry':
                        cv2.rectangle(image_cap_ori, (x, y), (x+w, y+h), (255,0,0), 4)
                        cv2.putText(image_cap_ori, 'angry', (int(float(x+w/2-43)), y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
                    elif emotion_text == 'sad':
                        cv2.rectangle(image_cap_ori, (x, y), (x+w, y+h), (0,0,255), 4)
                        cv2.putText(image_cap_ori, 'sad', (int(float(x+w/2-43)), y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                    elif emotion_text == 'happy':
                        cv2.rectangle(image_cap_ori, (x, y), (x+w, y+h), (255,255,0), 4)
                        cv2.putText(image_cap_ori, 'happy', (int(float(x+w/2-43)), y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,0), 1, cv2.LINE_AA)
                    elif emotion_text == 'surprise':
                        cv2.rectangle(image_cap_ori, (x, y), (x+w, y+h), (0,255,255), 4)
                        cv2.putText(image_cap_ori, 'surprise', (int(float(x+w/2-43)), y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,255), 1, cv2.LINE_AA)
                    elif emotion_text == 'disgust':
                        cv2.rectangle(image_cap_ori, (x, y), (x+w, y+h), (0,0,0), 4)
                        cv2.putText(image_cap_ori, 'disgust', (int(float(x+w/2-43)), y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                    elif emotion_text == 'fear':
                        cv2.rectangle(image_cap_ori, (x, y), (x+w, y+h), (255,0,255), 4)
                        cv2.putText(image_cap_ori, 'fear', (int(float(x+w/2-43)), y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255), 1, cv2.LINE_AA)
                    else:
                        cv2.rectangle(image_cap_ori, (x, y), (x+w, y+h), (0,255,0), 4)
                        cv2.putText(image_cap_ori, 'neutral', (int(float(x+w/2-43)), y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 1, cv2.LINE_AA)

                if save_info:
                    op_info_list = [ori_image_name,
                                    face_coordinates[0]*scaling_factor_x, face_coordinates[1]*scaling_factor_y,
                                    face_coordinates[2]*scaling_factor_x, face_coordinates[3]*scaling_factor_y]
                    for i in range(len(op_info_list)):
                        op_info_list[i] = str(op_info_list[i])
                    if detect_emo:
                        op_info_list.append(emotion_text)
                        for prob in emotion_prediction:
                            op_info_list.append(prob)
                    csv_writer.writerow(op_info_list)

            if save_image:
                cv2.imwrite(output_image_root + no_root_path + image_name + ".jpg", image_cap_ori)
            if save_info:
                csv_info.close()
            print(image_path+' DONE!!\tSpend Time: '+str(time.time()-st_time)+'(s)')

        else:
            os.makedirs(os.path.dirname(output_info_root + no_root_path), exist_ok=True)
            if os.path.isfile(output_info_root+no_root_path+spec_csv_name) == False:
                    first_construct_csv = True
            csv_info = codecs.open(output_info_root+no_root_path+spec_csv_name,
                                'a', encoding="utf_8_sig")
            csv_writer = csv.writer(csv_info)
            if first_construct_csv == True:
                csv_writer.writerow(info_name)
            err_msg = "The resolution of " + image_name + ".jpg is lower than 720p."
            csv_writer.writerow([err_msg])
            csv_info.close()

        src = image_path
        dst = in_finished_path + no_root_path + image_name + ".jpg"
        os.makedirs(os.path.dirname(in_finished_path + no_root_path), exist_ok=True)
        shutil.move(src, dst)
        if reduce_resolution == 1:
            image_ori_name = image_name[15:]
            os.remove(dst)
            os.rename(output_image_root+no_root_path+image_name+'.jpg', output_image_root+no_root_path+image_ori_name+'.jpg')

    shutil.rmtree(input_image_root, ignore_errors=True)
