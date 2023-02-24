import sys
import cv2
import torch.nn as nn
import torch
from torchvision import transforms, utils
import torchvision.models as models
from tqdm import tqdm
import json
import os 



category_index_dir = '/home/robert/Models/Bird_classification_Res18/checkpoint/Bird_I/category_index.json'
model_dir = '/home/robert/Models/Bird_classification_Res18/checkpoint/Bird_I/model.pth'


with open(category_index_dir,'r') as f:
    category_index = json.load(f)
resnet18 = models.resnet18(pretrained=True)
resnet18.fc =  nn.Sequential(resnet18.fc,nn.Dropout(p = 0.2),nn.Linear(in_features=1000, out_features=len(category_index), bias=True))
model = resnet18
model.load_state_dict(torch.load(model_dir))
model.eval()
model.cuda()
inference_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.CenterCrop((128,128)),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def classifier_inference(image_dir,detection_dir,out_dir):
    with open(detection_dir, 'r') as f:
            data = f.readlines()
    image = cv2.imread(image_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    w,h,c = image.shape
    y_preds = []
    for idx,line in enumerate(data):
        line = line.replace('\n','').split(' ')
        box = [max(0,int(i)) for i in line[2:]]
        cropped_image = image[min(box[1],w):min(w,box[3]),min(box[0],h):min(h,box[2]),:]
        inputs = inference_transform(cropped_image).unsqueeze(0).cuda()
        out = model(inputs)
        print (out)
        out = torch.max(out, dim=1)[1].squeeze().cpu().item()

        preds = category_index[str(out)]
        y_preds.append(preds)
    with open(detection_dir,'r') as f:
        data = f.readlines()
    file_name = os.path.basename(detection_dir)
    with open(os.path.join(out_dir,file_name),'w') as f:
        for i,line in enumerate(data):
            line = line.split(' ')
            line[0] = y_preds[i]
            f.writelines(','.join(line))


if __name__ == "__main__":
    image_dir = '/home/robert/Data/WaterFowl_Processed/drone_collection/I_DJI_0015.jpg'
    detection_dir ='/home/robert/Models/Retinanet_inference_example/results/40/detection-results/I_DJI_0015.txt'
    out_dir = './'
    classifier_inference(image_dir,detection_dir,out_dir)