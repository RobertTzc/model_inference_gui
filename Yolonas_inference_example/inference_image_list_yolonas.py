import super_gradients
import cv2
import torch
import torchvision.transforms as transforms
import os
import warnings
import glob
import os
import time
import json
import glob
import pandas as pd
from args import *
from tqdm import tqdm
import numpy as np

def get_sub_image(mega_image,overlap=0.2,ratio=1):
    #mage_image: original image
    #ratio: ratio * 512 counter the different heights of image taken
    #return: list of sub image and list fo the upper left corner of sub image
    coor_list = []
    sub_image_list = []
    w,h,c = mega_image.shape
    if w < 512 or h < 512:
        mega_image = image_padding(mega_image)
    size  = int(ratio*512)
    num_rows = int(w/int(size*(1-overlap)))
    num_cols = int(h/int(size*(1-overlap)))
    new_size = int(size*(1-overlap))
    for i in range(num_rows+1):
        if (i == num_rows):
            for j in range(num_cols+1):
                if (j==num_cols):
                    sub_image = mega_image[-size:,-size:,:]
                    coor_list.append([w-size,h-size])
                    sub_image_list.append (sub_image)
                else:
                    sub_image = mega_image[-size:,new_size*j:new_size*j+size,:]
                    coor_list.append([w-size,new_size*j])
                    sub_image_list.append (sub_image)
        else:
            for j in range(num_cols+1):
                if (j==num_cols):
                    sub_image = mega_image[new_size*i:new_size*i+size,-size:,:]
                    coor_list.append([new_size*i,h-size])
                    sub_image_list.append (sub_image)
                else:
                    sub_image = mega_image[new_size*i:new_size*i+size,new_size*j:new_size*j+size,:]
                    coor_list.append([new_size*i,new_size*j])
                    sub_image_list.append (sub_image)
    return sub_image_list,coor_list

def py_cpu_nms(dets, thresh):  
    """Pure Python NMS baseline.""" 
    dets = np.asarray(dets) 
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  
    scores = dets[:, 4] 
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    order = scores.argsort()[::-1]  
 
    keep = []  
    while order.size > 0:  
        i = order[0]  
        keep.append(i)  
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  
        inds = np.where(ovr <= thresh)[0]  
        order = order[inds + 1]  
    return keep

def get_image_height_model_4(image_name,image_height):
    if  image_height<=15:
        return image_height,0,15
    elif image_height<=30:
        return image_height,1,30
    elif image_height<=60:
        return image_height,2,60
    else:
        return image_height,3,90

def prepare_yolonas(model_dir):
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_name == 'cuda':
        return super_gradients.training.models.get('yolo_nas_m',num_classes=1,checkpoint_path=model_dir).cuda()
    else:
        return super_gradients.training.models.get('yolo_nas_m',num_classes=1,checkpoint_path=model_dir)

def inference_mega_image_Yolonas(image_list,model_root, image_out_dir,text_out_dir,scaleByAltitude, defaultAltitude=[],**kwargs):

    record = []
    model_15 = os.path.join(model_root,'ckpt_best15m.pth')
    model_30 = os.path.join(model_root,'ckpt_best30m.pth')
    model_60 = os.path.join(model_root,'ckpt_best60m.pth')
    model_90 = os.path.join(model_root,'ckpt_best90m.pth')
    model_list = [model_15,model_30,model_60,model_90]
    net_list = []
    for model_dir in model_list:
        net_list.append(prepare_yolonas(model_dir))

    with tqdm(total = len(image_list)) as pbar:
        for idxs, image_dir in (enumerate(image_list)):
            pbar.update(1)
            start_time = time.time()
            image_name = os.path.split(image_dir)[-1]
            mega_image = cv2.imread(image_dir)
            ratio = 1
            bbox_list = []
            sub_image_list, coor_list = get_sub_image(mega_image, overlap=0.2, ratio=ratio)
            for index, sub_image in enumerate(sub_image_list):
                if scaleByAltitude:
                    image_taken_height,model_index,model_height = get_image_height_model_4(image_name.split('.')[0],int(defaultAltitude[idxs]))
                    ratio = round(model_height/image_taken_height, 2)
                    selected_model = net_list[model_index]
                sub_image_dir = './tmp.JPG'
                cv2.imwrite(sub_image_dir,sub_image)
                images_predictions = selected_model.predict(sub_image_dir)
                os.remove(sub_image_dir)
                image_prediction = next(iter(images_predictions))
                labels = image_prediction.prediction.labels
                confidences = image_prediction.prediction.confidence
                bboxes = image_prediction.prediction.bboxes_xyxy

                for i in range(len(labels)):
                    label = labels[i]
                    confidence = confidences[i]
                    bbox = bboxes[i]
                    bbox_list.append([coor_list[index][1]+bbox[0], coor_list[index][0]+bbox[1],coor_list[index][1]+bbox[2], coor_list[index][0]+bbox[3], confidence])
            if (len(bbox_list) != 0):
                bbox_list = np.asarray([box for box in bbox_list])
                box_idx = py_cpu_nms(bbox_list, 0.25)
                selected_bbox = bbox_list[box_idx]
                selected_bbox = sorted(selected_bbox,key = lambda x: x[4],reverse = True)

            txt_name = os.path.basename(image_dir).split('.')[0]+'.txt'
            with open(os.path.join(text_out_dir,txt_name), 'w') as f:
                if (len(selected_bbox) != 0):
                    for box in selected_bbox:
                        f.writelines('bird,{},{},{},{},{}\n'.format(
                            box[4], int(box[0]), int(box[1]), int(box[2]), int(box[3])))
            try:
                re = read_LatLotAlt(image_dir)
            except:
                re = {'latitude':0.0,
                      'longitude':0.0,
                      'altitude':0.0}
            record.append([os.path.basename(image_dir),kwargs['date_list'][idxs],kwargs['location_list'][idxs],
                       defaultAltitude[idxs],re['latitude'],re['longitude'],re['altitude'],
                       len(selected_bbox),round(time.time()-start_time,2)])
    record = pd.DataFrame(record)
    record.to_csv(kwargs['csv_out_dir'],header = ['image_name','date','location','altitude','latitude_meta','longitude_meta','altitude_meta','num_birds','time_spent(sec)'],index = True)