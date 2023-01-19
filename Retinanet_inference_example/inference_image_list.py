import cv2
import torch
import torchvision.transforms as transforms
from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw, ImageFont
import os
import warnings
import glob
import os
import numpy as np
from PIL import Image
import numpy as np
import time
import json
import sys
import glob
import pandas as pd
from args import *
from pyexiv2 import Image
from utils import read_LatLotAlt,get_GSD
from WaterFowlTools.mAp import mAp_calculate,plot_f1_score,plot_mAp
import matplotlib.pyplot as plt
from WaterFowlTools.utils import py_cpu_nms, get_image_taking_conditions, get_sub_image
from classifier import Classifier
from visualize import plot_results
warnings.filterwarnings("ignore")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
model_conf_threshold = {'Bird_A':0.2,'Bird_B':0.2,'Bird_C':0.2,'Bird_D':0.2,'Bird_E':0.2,'Bird_drone':0.22}
model_extension = {'Bird_drone':{40:('_alt_30',30),75:('_alt_60',60),90:('_alt_90',90)}}

def get_model_conf_threshold (model_type):
    if (model_type in model_conf_threshold):
        return model_conf_threshold[model_type]
    else:
        return 0.3

def get_model_extension(model_type,model_dir,defaultaltitude):
    if(model_type in model_extension):
        model_ext = model_extension[model_type]
        for altitude_thresh in model_ext:
            if (altitude_thresh>=defaultaltitude):
                ref_altitude = model_ext[altitude_thresh][1]
                model_dir = model_dir.replace('.pkl',model_ext[altitude_thresh][0]+'.pkl')
                return model_dir,ref_altitude
        model_dir = model_dir.replace('.pkl',model_ext[max(model_ext.keys())][0]+'.pkl')
        return model_dir,model_ext[max(model_ext.keys())][1]
    else:
        return model_dir,90
     
def inference_mega_image_Retinanet(image_list, model_dir, image_out_dir,text_out_dir, visualize,scaleByAltitude=True, defaultAltitude=[],**kwargs):
    model_type = kwargs['model_type']
    conf_thresh = get_model_conf_threshold(model_type=model_type)
    model_dir,ref_altitude = get_model_extension(model_type=model_type,model_dir=model_dir,defaultaltitude=defaultAltitude[0])
    print ('The model actually used: ',model_dir)
    if (kwargs['device']!=torch.device('cuda')):
        print ('loading CPU mode')
        device = torch.device('cpu')
        net = torch.load(model_dir,map_location=device)
        net = net.module.to(device)
    else:
        device = torch.device('cuda')
        net = torch.load(model_dir)
    net.to(device)
    print('check net mode',next(net.parameters()).device)
    encoder = DataEncoder(device)
    record = []
    for idxs, image_dir in (enumerate(image_list)):
        start_time = time.time()
        # try:
        #     altitude = get_image_taking_conditions(image_dir)['altitude']
        #     print ('Processing image name: {} with Altitude of {}'.format(os.path.basename(image_dir),altitude))
        # except:
        altitude = int(defaultAltitude[idxs])
        print ('Using default altitude for: {} with Altitude of {}'.format(os.path.basename(image_dir),altitude))
        if scaleByAltitude:
            GSD,ref_GSD = get_GSD(altitude,camera_type='Pro2', ref_altitude=ref_altitude) # Mavic2 Pro GSD equations
            ratio = 1.0*ref_GSD/GSD
        else:
            ratio = 1.0
        print('Processing scale {}'.format(ratio))
        bbox_list = []
        mega_image = cv2.imread(image_dir)
        mega_image = cv2.cvtColor(mega_image, cv2.COLOR_BGR2RGB)
        sub_image_list, coor_list = get_sub_image(
            mega_image, overlap=0.2, ratio=ratio)
        for index, sub_image in enumerate(sub_image_list):
            with torch.no_grad():
                inputs = transform(cv2.resize(
                    sub_image, (512, 512), interpolation=cv2.INTER_AREA))
                inputs = inputs.unsqueeze(0).to(kwargs['device'])
                loc_preds, cls_preds = net(inputs)
                boxes, labels, scores = encoder.decode(
                    loc_preds.data.squeeze(), cls_preds.data.squeeze(), 512, CLS_THRESH = conf_thresh,NMS_THRESH = 0.5)
            if (len(boxes.shape) != 1):
                for idx in range(boxes.shape[0]):
                    x1, y1, x2, y2 = list(
                        boxes[idx].cpu().numpy())  # (x1,y1, x2,y2)
                    score = scores.cpu().numpy()[idx]
                    bbox_list.append([coor_list[index][1]+ratio*x1, coor_list[index][0]+ratio*y1,
                                     coor_list[index][1]+ratio*x2, coor_list[index][0]+ratio*y2, score])
        txt_name = os.path.basename(image_dir).split('.')[0]+'.txt'
        num_bird = 0
        with open(os.path.join(text_out_dir,txt_name), 'w') as f:
            if (len(bbox_list) != 0):
                bbox_list = np.asarray([box for box in bbox_list])
                box_idx = py_cpu_nms(bbox_list, 0.25)
                num_bird = len(box_idx)
                selected_bbox = bbox_list[box_idx]
                print('Finished on {},\tfound {} birds'.format(
                os.path.basename(image_dir), len(selected_bbox)))
                selected_bbox = sorted(selected_bbox,key = lambda x: x[4],reverse = True)
                for box in selected_bbox:
                    f.writelines('bird,{},{},{},{},{}\n'.format(
                        box[4], int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                    if (visualize):
                        cv2.putText(mega_image, str(round(box[4], 2)), (int(box[0]), int(
                            box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.rectangle(mega_image, (int(box[0]), int(
                            box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
        mega_image = cv2.cvtColor(mega_image, cv2.COLOR_RGB2BGR)
        if (visualize):
            cv2.imwrite(os.path.join(image_out_dir,os.path.basename(image_dir)), mega_image)
        try:
            re = read_LatLotAlt(image_dir)
        except:
            re = {'latitude':0.0,
                  'longitude':0.0,
                  'altitude':0.0}
        record.append([os.path.basename(image_dir),kwargs['date_list'][idxs],kwargs['location_list'][idxs],
                       defaultAltitude[idxs],re['latitude'],re['longitude'],re['altitude'],
                       num_bird,round(time.time()-start_time,2)])
    record = pd.DataFrame(record)
    record.to_csv(kwargs['csv_out_dir'],header = ['image_name','date','location','altitude','latitude_meta','longitude_meta','altitude_meta','num_birds','time_spent(sec)'],index = True)
    
        



if __name__ == '__main__':
    args = get_args()
    model_type = args.model_type
    image_list = glob.glob(os.path.join(args.image_root,'*.{}'.format(args.image_ext)))
    image_name_list = [os.path.basename(i) for i in image_list]

    altitude_list = [args.image_altitude for _ in image_list]
    
    location_list = [args.image_location for _ in image_list]
    date_list = [args.image_date for _ in image_list]
    
    target_dir = args.out_dir
    model_dir = args.model_dir
    image_out_dir = os.path.join(target_dir,'visualize-results')
    text_out_dir = os.path.join(target_dir,'detection-results')
    csv_out_dir = os.path.join(target_dir,'detection_summary.csv')
    print ('*'*30)
    print ('Using model type: {}'.format(model_type))
    print ('Using device: {}'.format(device))
    print ('Image out dir: {}'.format(image_out_dir))
    print ('Texting out dir: {}'.format(text_out_dir))
    print ('Inferencing on Images:\n {}'.format(image_list))
    print ('Altitude of each image:\n {}'.format(altitude_list))
    print ('Visualize on each image:\n {}'.format(args.visualize))
    print ('*'*30)
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(text_out_dir, exist_ok=True)
    inference_mega_image_Retinanet(
		image_list=image_list, model_dir = model_dir, image_out_dir = image_out_dir,text_out_dir = text_out_dir,csv_out_dir = csv_out_dir,
		scaleByAltitude=args.use_altitude, defaultAltitude=altitude_list,date_list = date_list,location_list =location_list,
        visualize = args.visualize,device = device,model_type = model_type)


    if (args.classification):
        tmp_cropped_dir = './tmp'
        batch_size = 8
        classifier = Classifier(tmp_cropped_dir, 8, text_out_dir,args.classification_weight,device)
        for image_dir in image_list:
            image_name = os.path.basename(image_dir)
            detection_dir  = os.path.join(text_out_dir,image_name.split('.')[0]+'.txt')
            classifier.inference(image_dir,detection_dir)
            plot_results(image_dir,detection_dir,image_out_dir+'_final')


    if (args.evaluate):
        try:
            precision, recall, sum_AP, mrec, mprec, area = mAp_calculate(image_name_list = image_name_list, 
                                                                        gt_txt_list=[os.path.splitext(i)[0]+'.txt' for i in image_list],
                                                                        pred_txt_list = [text_out_dir+'/'+os.path.splitext(i)[0]+'.txt' for i in image_name_list],
                                                                        iou_thresh=0.3)
            plot_f1_score(precision, recall, args.model_type, text_out_dir, area, 'f1_score', color='r')
            plt.legend()
            plt.savefig(os.path.join(target_dir,'f1_score.jpg'))
            plt.figure()
            plot_mAp(precision, recall, mprec, mrec,  args.model_type, area, 'mAp', color='r')
            plt.legend()
            plt.savefig(os.path.join(target_dir,'mAp.jpg'))
            print('Evaluation completed, proceed to wrap result')
        except:
            print('Failed to evaluate, Skipped')
    argparse_dict = vars(args)
    with open(os.path.join(target_dir,'configs.json'),'w') as f:
        json.dump(argparse_dict,f,indent=4)
     
