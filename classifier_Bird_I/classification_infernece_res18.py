import sys
import torch.nn as nn
import cv2
import torch
from torchvision import transforms, utils
import torchvision.models as models
from tqdm import tqdm
import json
import glob
import os

def plot_box(image_dir,txt_dir,out_dir):
    image = cv2.imread(image_dir)
    image_name = os.path.basename(image_dir)
    os.makedirs(out_dir,exist_ok=True)
    with open(txt_dir,'r') as f:
        data = f.readlines()
    for line in data:
        line = line.replace('\n','').split(',')
        n = line[0]+'_{}'.format(round(float(line[1]),2))
        box = line[2:]+[n]
        cv2.putText(image, n, (int(box[0]), int(
                                box[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        cv2.rectangle(image, (int(box[0]), int(
            box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(out_dir,image_name),image)

def res18_classifier_inference(model_dir,category_index_dir,image_list,detection_root_dir,text_out_dir,visual_out_dir,device):
    os.makedirs(text_out_dir,exist_ok=True)
    test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop((128,128)),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    with open(category_index_dir,'r') as f:
        category_dict = json.load(f)
    category_list = list(category_dict.values())

    resnet18 = models.resnet18(pretrained=True)
    resnet18.fc =  nn.Sequential(resnet18.fc,nn.Dropout(p = 0.2),nn.Linear(in_features=1000, out_features=len(category_dict), bias=True))
    resnet18.load_state_dict(torch.load(model_dir))
    resnet18 = resnet18.eval()
    resnet18.to(device)
    resnet18.eval()


    for image_dir in image_list:
        file_name = os.path.basename(image_dir)
        pd_dir = os.path.join(detection_root_dir,file_name.split('.')[0]+'.txt')
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        with open(pd_dir,'r') as f:
            data = f.readlines()
        pred_data = []
        for line in data:
            line = line.replace('\n','').split(',')
            coord = [int(i) for i in line[2:]]
            coord= [max(0,coord[0]),max(0,coord[1]),min(image.shape[1],coord[2]),min(image.shape[0],coord[3]),]
            cropped_image = image[coord[1]:coord[3],coord[0]:coord[2],:]
            inputs = test_transform(cropped_image).unsqueeze(0).to(device)
            out = resnet18(inputs)
            out = torch.max(out, dim=1)[1].squeeze().data
            preds = category_list[out]
            pred_data.append([preds,line[1],line[2],line[3],line[4],line[5]])
        with open(text_out_dir+'/{}.txt'.format(file_name.split('.')[0]),'w') as f:
            for line in pred_data:
                line = ','.join(line)
                f.writelines(line+'\n')
        plot_box(image_dir,text_out_dir+'/{}.txt'.format(file_name.split('.')[0]),visual_out_dir)



if __name__ =='__main__':
    model_folder_name = 'Bird_I_classifier'
    fix_size = False

    model_dir = './checkpoint/{}/model.pth'.format(model_folder_name)
    category_index_dir = './checkpoint/{}/category_index.json'.format(model_folder_name)
    image_root = '/home/robert/Data/Bird_I_Test'
    for folder_dir in glob.glob('/home/robert/Models/model_inference_gui/Retinanet_inference_example/result/Bird_I_Test/*'):
        print (folder_dir)
        folder_name = folder_dir.split('/')[-1]
        image_root_dir = image_root+'/'+folder_name
        detection_root_dir = folder_dir+'/detection-results'
        target_root_dir = './checkpoint/{}/Bird_I_Test/{}/classification_result'.format(model_folder_name,folder_name)
        inference_by_folder(model_dir,category_index_dir,image_root_dir,detection_root_dir,target_root_dir,fix_size = fix_size)