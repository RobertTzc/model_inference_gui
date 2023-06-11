'''
The script designed to evaluate the classification result
Across all the detetcions, if the prediction has IOU>0.2 with a GT,
    consider they are same target birds(if the detection has IOU>0.2 with multiple GT's, take the highest IOU one as reference)
If a detection has no GT with it, consider the label as not a bird as valid label
The result will be for each prediction class: Num of correct prediction/Num of total prediction in this category
'''
import glob
import numpy as np
import os
import cv2
class_pool= ['Ring-necked duck Male', 'American Widgeon_Female', 'Ring-necked duck Female', 'Canvasback_Male',
              'Canvasback_Female', 'Scaup_Male',
              'Shoveler_Male', 'Not a bird', 'Shoveler_Female', 'Gadwall', 'Unknown', 'Mallard Male', 'Pintail_Male', 'Green-winged teal',
              'White-fronted Goose', 'Snow/Ross Goose (blue)', 'Snow/Ross Goose', 'Mallard Female', 'Coot', 'Pelican', 'American Widgeon_Male',
              'Canada Goose']
class_dict = dict()
for c in class_pool:
    class_dict[c] = [0,0]
class_dict['fp'] = [0,0]

def match_category(pred_box,gt_bbox):
    re = []
    for gt_box in gt_bbox:
        re.append(IoU2(pred_box,gt_box))
    if (max(re)<0.1):
        return 'no match'
    else:
        return gt_bbox[re.index(max(re))][-1]

def IoU2(pred_box, true_box):
    bb = pred_box[:4]
    bbgt = true_box[:4]
    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]),
          min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1
    if iw > 0 and ih > 0:
        # compute overlap (IoU) = area of intersection / area of union
        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                          + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
        ov = iw * ih / ua
        return ov
    return 0.0

def plot_gt(image_dir,txt_dir,out_dir):
    image = cv2.imread(image_dir)
    image_name = os.path.basename(image_dir)
    os.makedirs(out_dir,exist_ok=True)
    with open(txt_dir,'r') as f:
        data = f.readlines()
    for line in data:
        line = line.replace('\n','').split(',')
        n = line[1]
        box = line[2:]+[n]
        cv2.putText(image, n, (int(box[0]), int(
                                box[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        cv2.rectangle(image, (int(box[0]), int(
            box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(out_dir,image_name),image)

gt_root_dir = './example_images/Bird_I'
gt_folder_list = glob.glob(os.path.join(gt_root_dir,'test'))


for gt_folder_dir in gt_folder_list:
    pd_folder_dir = './Result/0410/{}/res18/{}/detection-results'
    #print (pd_folder_dir)
    for gt_txt in glob.glob(os.path.join(gt_folder_dir,'*_class.txt')):
        try:
            image_dir = gt_txt.replace('_class.txt','.jpg')
            plot_gt(image_dir,gt_txt,gt_folder_dir+'_visual')
        except:
            image_dir = gt_txt.replace('_class.txt','.JPG')
            plot_gt(image_dir,gt_txt,gt_folder_dir+'_visual')
        pd_txt = os.path.join(pd_folder_dir,os.path.basename(gt_txt).replace('_class',''))
        with open(gt_txt,'r')as f:
            data = f.readlines()
        gt_bbox = []
        for line in data:
            line = line.replace('\n','').split(',')
            box = [int(i) for i in line[2:]]
            gt_bbox.append(box+[line[1]])
            print (box)
        with open(pd_txt,'r')as f:
            data = f.readlines()
        pd_bbox = []
        for line in data:
            line = line.replace('\n','').split(',')
            pd_bbox.append([int(i) for i in line[2:]]+[line[0]])
        for pd_box in pd_bbox:
            class_dict[pd_box[-1]][1]+=1
            if (match_category(pd_box,gt_bbox)==pd_box[-1]):
                class_dict[pd_box[-1]][0]+=1
            else:
                print (match_category(pd_box,gt_bbox),'*'*10,pd_box[-1])
            if (match_category(pd_box,gt_bbox)=='no match'):
                class_dict['fp'][1]+=1



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def overlapped_bar(df, show=False, width=0.9, alpha=.5,
                   title='', xlabel='', ylabel='', **plot_kwargs):
    """Like a stacked bar chart except bars on top of each other with transparency"""
    xlabel = xlabel or df.index.name
    N = len(df)
    M = len(df.columns)
    indices = np.arange(N)
    colors = ['steelblue', 'firebrick', 'darksage', 'goldenrod', 'gray'] * int(M / 5. + 1)
    for i, label, color in zip(range(M), df.columns, colors):
        kwargs = plot_kwargs
        kwargs.update({'color': color, 'label': label})
        plt.bar(indices, df[label], width=width, alpha=alpha if i else 1, **kwargs)
        plt.xticks(indices + .5 * width,
                   ['{}'.format(idx) for idx in df.index.values])
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show:
        plt.show()
    return plt.gcf()

high   = [i[1] for i in class_dict.values()]
low   = [i[0] for i in class_dict.values()]

df = pd.DataFrame(np.matrix([high, low]).T, columns=['total predictions','correct predictions'],
                  index=pd.Index(['T%s' %i for i in range(len(high))],
                  name='Index'))
overlapped_bar(df, show=True)       

print ([(i,j) for i,j in enumerate(class_dict.keys())])