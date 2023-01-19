#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 17:11:37 2022

@author: shiqi
"""


from fractions import gcd
from absl import flags
import json
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import os
import torch
import cv2
import collections
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from resnet_pytorch import ResNet
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import collections
import shutil
from absl import app


class Classifier:

    def __init__(self, tmp_cropped_dir, batch_size, out_dir,weight_dir,device):
        
        self.tmp_cropped_dir = tmp_cropped_dir
        self.batch_size = batch_size
        self.out_dir = out_dir
        self.weight_dir = weight_dir
        self.device = device

    def crop_detection(self,image_dir,detection_dir):
        with open(detection_dir, 'r') as f:
            data = f.readlines()
        idx = 0

        # check if directory exists:
        if os.path.exists(self.tmp_cropped_dir):
            shutil.rmtree(self.tmp_cropped_dir)
        os.makedirs(self.tmp_cropped_dir,exist_ok=True)
        os.makedirs(os.path.join(self.tmp_cropped_dir,'images'),exist_ok=True)
        


        image = cv2.imread(image_dir)
        w,h,c = image.shape
        for idx,line in enumerate(data):
            line = line.replace('\n','').split(',')
            box = [max(0,int(i)) for i in line[2:]]
            cropped_image = image[min(box[1],w):min(w,box[3]),min(box[0],h):min(h,box[2]),:]
            cv2.imwrite(os.path.join(self.tmp_cropped_dir,'images','{}.jpg'.format(idx)),cropped_image)



    def inference(self,image_dir,detection_dir):
        self.crop_detection(image_dir,detection_dir)
        number_class = 22
        pretrained_size = (128, 128)
        pretrained_means = [0.485, 0.456, 0.406]
        pretrained_stds = [0.229, 0.224, 0.225]

        class_index = ['Ring-necked duck Male', 'American Widgeon_Female', 'Ring-necked duck Female', 'Canvasback_Male',
              'Canvasback_Female', 'Scaup_Male',
              'Shoveler_Male', 'Not a bird', 'Shoveler_Female', 'Gadwall', 'Unknown', 'Mallard Male', 'Pintail_Male', 'Green-winged teal',
              'White-fronted Goose', 'Snow/Ross Goose (blue)', 'Snow/Ross Goose', 'Mallard Female', 'Coot', 'Pelican', 'American Widgeon_Male',
              'Canada Goose']

        test_transforms = transforms.Compose([
            transforms.Resize(pretrained_size),
            transforms.CenterCrop(pretrained_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
        ])

        test_iterator = torch.utils.data.DataLoader(
            datasets.ImageFolder(self.tmp_cropped_dir, test_transforms),
            batch_size=self.batch_size)

        model = ResNet.from_pretrained('resnet18')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, number_class)

        checkpoint = torch.load(self.weight_dir)
        model.load_state_dict(checkpoint)
        model.eval()
        model = model.to(self.device)
        y_preds = []

        with torch.no_grad():
            for (x, y) in tqdm(test_iterator):
                x = x.to(self.device)
                y_pred = model(x)

                output = (torch.max(torch.exp(y_pred), 1)
                          [1]).data.cpu().numpy()
                y_preds.extend(output)

        preds = [class_index[i] for i in y_preds]

        with open(detection_dir,'r') as f:
            data = f.readlines()
        file_name = os.path.basename(detection_dir)
        with open(os.path.join(self.out_dir,file_name),'w') as f:
            for i,line in enumerate(data):
                line = line.split(',')
                line[0] = preds[i]
                f.writelines(','.join(line))
        
if __name__ == '__main__':
        
    tmp_cropped_dir = './tmp'
    batch_size = 8
    out_dir = '/home/zt253/Models/Retinanet_inference_example/results/detection-results'
    weight_dir = '/home/zt253/Downloads/Zhangyang 1/Zhangyang/resnet18-sklearn-sf-vr-last.pt'
    device = torch.device('cpu')
    classifier = Classifier(tmp_cropped_dir, batch_size, out_dir,weight_dir,device)
    classifier.inference('/home/zt253/Models/Retinanet_inference_example/example_images/Bird_drone/15m/I_DJI_0015.jpg',
    '/home/zt253/Models/Retinanet_inference_example/results/detection-results/I_DJI_0015.txt')
