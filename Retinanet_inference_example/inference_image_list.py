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
from utils import read_LatLotAlt
from WaterFowlTools.mAp import mAp_calculate,plot_f1_score,plot_mAp
import matplotlib.pyplot as plt
from retinanet_inference_ver3 import Retinanet_instance
warnings.filterwarnings("ignore")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def inference_mega_image_Retinanet(image_list, model_dir, image_out_dir,text_out_dir,device,scaleByAltitude, defaultAltitude=[],**kwargs):
    model_type = kwargs['model_type']
    if (model_type=='Bird_drone_KNN'):
        load_w_config = True
    else:
        load_w_config = False
    model = Retinanet_instance(transform,model_type,model_dir,device,load_w_config,int(defaultAltitude[0]))
    record = []
    for idxs, image_dir in (enumerate(image_list)):
        start_time = time.time()
        mega_image,bbox_list = model.inference(image_dir,0.2,scaleByAltitude)
        txt_name = os.path.basename(image_dir).split('.')[0]+'.txt'
        num_bird = 0
        with open(os.path.join(text_out_dir,txt_name), 'w') as f:
            if (len(bbox_list) != 0):
                print('Finished on {},\t Found {} birds'.format(
                os.path.basename(image_dir), len(bbox_list)))
                bbox_list = sorted(bbox_list,key = lambda x: x[4],reverse = True)
                for box in bbox_list:
                    f.writelines('bird,{},{},{},{},{}\n'.format(
                        box[4], int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        mega_image = cv2.cvtColor(mega_image,cv2.COLOR_RGB2BGR)
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
    print (image_list)

    altitude_list = [args.image_altitude for _ in image_list]
    
    location_list = [args.image_location for _ in image_list]
    date_list = [args.image_date for _ in image_list]
    
    target_dir = args.out_dir
    model_dir = args.model_dir
    image_out_dir = os.path.join(target_dir,'visualize-results')
    text_out_dir = os.path.join(target_dir,'detection-results')
    csv_out_dir = os.path.join(target_dir,'detection_summary.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
     