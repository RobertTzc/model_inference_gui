import sys
sys.path.insert(1,'/home/zt253/Models/retinanet_UnionData')
import cv2
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw,ImageFont
from PIL.ExifTags import TAGS
import time
import warnings
import glob 
import os
import numpy as np
from PIL import Image
import math
import tqdm

warnings.filterwarnings("ignore")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

def GSD_calculation(height = 90.0,drone_type = 'Pro2'):
    if (drone_type == 'Pro2'):
        GSD=(13.2 * height)/(10.26*5472)
    elif (drone_type == 'Air2'):
        GSD = (6.4*height)/(4.3*8000)
    else:
        GSD=(13.2 * height)/(10.26*5472)
    return GSD


def read_gps(image_name,info_dir):
	info_txt = open(info_dir,'r')
	for line in info_txt.readlines():
		img_name,_,coor_lat,coor_lont,altitude = line.split(' ')
		if (img_name in image_name):
			coor_lat = float(coor_lat.replace('(','').replace(',',''))
			coor_lont = float(coor_lont.replace('(','').replace(',',''))
			altitude = float(altitude.split(')')[0])
			return coor_lat, coor_lont, altitude

def get_sub_image(mega_image,overlap=0.2,ratio=1):
	#mage_image: original image
	#ratio: ratio * 512 counter the different heights of image taken
	#return: list of sub image and list fo the upper left corner of sub image
	if (mega_image.shape[0] ==512):
		return [mega_image],[[0,0]]
	coor_list = []
	sub_image_list = []
	w,h,c = mega_image.shape
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


def get_height(image_dir):
	image = Image.open(image_dir)
	try:
		exifdata = image._getexif()
		for tag_id in exifdata:
		    # get the tag name, instead of human unreadable tag id
		    tag = TAGS.get(tag_id, tag_id)
		    data = exifdata.get(tag_id)
		    # decode bytes 
		    if (tag=='GPSInfo'):
		    	return (data[6][0])
	except:
		data = 90
	return 90
def inference(model_dir,image_list,isHeight):
	net = RetinaNet(num_classes=1)
	net= torch.load(model_dir).cuda()
	get_height(image_list[0])
	height_list = [get_height(image_dir) for image_dir in image_list]

	encoder = DataEncoder()
	result_bbox = dict()
	for idxs,(height,image_dir) in tqdm.tqdm(enumerate(zip(height_list,image_list))):
		bbox_list = []
		mega_image  = cv2.imread(image_dir)
		image_name = image_dir.split('/')[-1]
		mega_image = cv2.cvtColor(mega_image,cv2.COLOR_BGR2RGB)
		if isHeight and height!=90:
			ratio = GSD_calculation(height,'Pro2')/GSD_calculation(90.0,'Pro2')
			#print ((0.004684 *height**2 - 1.05*height + 72.11)/17,height)
		else:
			ratio = 1.0
		sub_image_list,coor_list = get_sub_image(mega_image,overlap = 0.2,ratio =ratio)
		for index,sub_image in enumerate(sub_image_list):
			with torch.no_grad():
				inputs = transform(cv2.resize(sub_image,(512,512),interpolation = cv2.INTER_AREA))
				inputs = inputs.unsqueeze(0).cuda()
				loc_preds, cls_preds = net(inputs)
				boxes, labels, score = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(),512,0.1,0.1)
			if (len(boxes.shape)!=1):
				for idx in range(boxes.shape[0]):
					x1,y1,x2,y2 = list(boxes[idx].cpu().numpy()) # (x1,y1, x2,y2)
					s = score[idx]
					bbox_list.append([score.cpu().numpy()[idx],coor_list[index][1]+ratio*x1, coor_list[index][0]+ratio*y1, coor_list[index][1]+ratio*x2, coor_list[index][0]+ratio*y2])
		bbox_list = np.asarray(bbox_list)
		box_idx = py_cpu_nms(bbox_list,0.2)	
		#print (bbox_list[box_idx])
		result_bbox[image_name] = bbox_list[box_idx]

	return result_bbox
if __name__ == '__main__':
	re =  (inference(model_dir='/home/zhicheng/Models/retinanet_UnionData/checkpoint/2019_Summer_decoy_PNratio0.2/final_model.pkl',image_list = ['/home/zhicheng/Data/Union_Data/Eagle_bluff_sep29/DJI_0117.JPG'],isHeight=True))


