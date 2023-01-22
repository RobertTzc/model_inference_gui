import sys
sys.path.insert(0,'Retinanet_inference_example')
from Retinanet_inference_example.inference_image_list import inference_mega_image_Retinanet

from WaterFowlTools.utils import py_cpu_nms
from create_category_list_json import *
from tkinter import filedialog, dialog, Radiobutton
import tkinter as tk
from tkinter import *
import os
import os.path as path
import glob
from PIL import Image, ImageTk
from PIL import Image, ImageDraw,ImageFont
import collections
import torch

class ClassifyGUI():
    def __init__(self, config_data, root):
        self.root = root
        self.config = config_data
        self.root.geometry(
            str(self.config['GUIResolution'][0])+'x'+str(self.config['GUIResolution'][1]))
        self.image_preview_size = [int(self.config['RelativeLayoutImageView'][0]*self.config['GUIResolution'][0]), int(
            self.config['RelativeLayoutImageView'][1]*self.config['GUIResolution'][1])]
        self.image_list = []
        self.out_dir = ''
        self.image_id = 0
        self.detection_model_type = tk.StringVar()
        self.classification_model_type = tk.StringVar()
        self.altitude = 15
        self.detection_model_dir = ''
        self.classification_model_dir = ''
        self.detection_boxes = dict()
        self.confidence_threshold = 10
        self.NMS_threshold = 10
        self.bbox = []
        self.filtered_box_idx = []
        self.config_UI()

    def config_UI(self):
        # Menu bar configuration
        menubar = Menu(self.root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="open_single_image",
                             command=self.open_single_image)
        filemenu.add_command(label="open_image_dir",
                             command=self.open_image_dir)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=about)
        menubar.add_cascade(label="Help", menu=helpmenu)

        # Image preview configuration
        self.root.config(menu=menubar)
        self.image_preview = Image.new(
            'RGB', (self.image_preview_size[0], self.image_preview_size[1]))
        self.image_preview_tk = ImageTk.PhotoImage(self.image_preview)
        self.image_preview_window = Label(
            root, image=self.image_preview_tk, width=self.image_preview_size[0], height=self.image_preview_size[1])
        self.image_preview_window .grid(
            row=0, column=0, rowspan=20, columnspan=2, sticky=W+E+N+S)

        # Model type selection
        Label(self.root,text="Detection Model selection").grid(row=1, column=3, columnspan=3, sticky=W)
        Radiobutton(self.root, text='Retina-Net', variable=self.detection_model_type, value='Retina-Net',state=ACTIVE).grid(row=2, column=3, sticky=W)
        Radiobutton(self.root, text='Faster-RCNN', variable=self.detection_model_type, value='Faster-RCNN',state=DISABLED).grid(row=2, column=4, sticky=W)
        Radiobutton(self.root, text='YOLOv5', variable=self.detection_model_type, value='YOLOv5',state=DISABLED).grid(row=2, column=5, sticky=W)
        self.detection_model_type.set("Retina-Net")

        Label(self.root,text="Classification Model selection").grid(row=3, column=3, columnspan=3, sticky=W)
        Radiobutton(self.root, text='Res18', variable=self.classification_model_type, value='Res18',state=DISABLED).grid(row=4, column=3, sticky=W)
        Radiobutton(self.root, text='xxx', variable=self.classification_model_type, value='Faster-RCNN',state=DISABLED).grid(row=4, column=4, sticky=W)
        Radiobutton(self.root, text='xxx', variable=self.classification_model_type, value='YOLOv5',state=DISABLED).grid(row=4, column=5, sticky=W)

        #Other parameters:
        Label(self.root,text="Image Altitude",).grid(row=5, column=3, columnspan=2, sticky=W)
        self.altitude_entry = Entry(self.root)
        self.altitude_entry.grid(row=5, column=5,columnspan=4, sticky=W)
        self.altitude_entry.insert(0,'Input altitude')

        Label(self.root,text="Detection Model dir",).grid(row=6, column=3, columnspan=2, sticky=W)
        Button(root, height=4, text="Load Model",
            command=self.load_detection_model, fg='black').grid(row=6, column=5,sticky=W)
        self.detection_model_info_label = Label(self.root,text='')
        self.detection_model_info_label.grid(row=7, column=4, columnspan=5,rowspan = 2,sticky=W+N)
        

        Label(self.root,text="Classification Model dir",).grid(row=9, column=3, columnspan=2, sticky=W)
        Button(root, height=4, text="Load Model",
            command=self.load_classification_model, fg='black').grid(row=9, column=5,sticky=W)
        self.classification_model_info_label = Label(self.root,text='')
        self.classification_model_info_label.grid(row=10, column=4, columnspan=5,sticky=W+N)
        
        

        #Slide bar for confidence and NMS control.
        Label(self.root, text="confidence_threshold(%)").grid(
            row=11, column=3, columnspan=2, sticky=W)
        self.confidence_threshold_slider = Scale(
            self.root, from_=10, to=100, orient=HORIZONTAL, length=200)
        self.confidence_threshold_slider.bind(
            "<ButtonRelease-1>", self.update_confidence_threshold)
        self.confidence_threshold_slider.grid(row=11, column=5,columnspan=5,sticky=W)
        self.confidence_threshold_slider.set(self.confidence_threshold)

        Label(self.root, text="NMS_threshold(%)").grid(
            row=12, column=3, columnspan=2, sticky=W)
        self.NMS_threshold_slider = Scale(
            self.root, from_=10, to=100, orient=HORIZONTAL, length=200)
        self.NMS_threshold_slider.bind("<ButtonRelease-1>", self.update_NMS)
        self.NMS_threshold_slider.grid(row=12, column=5, columnspan=5,sticky=W)
        self.NMS_threshold_slider.set(self.NMS_threshold)

        self.inference_info_label = Label(self.root, text='inference information', anchor='w')
        self.inference_info_label.grid(row=13, column=3, columnspan=3,sticky=W+N+E+S)

        Label(self.root,text = 'Date collected', anchor='w').grid(row=16, column=3,sticky=W+N+E+S)
        self.date_info = Entry()
        self.date_info.grid(row=16, column=5, columnspan=4,sticky=W)

        Label(self.root,text = 'location', anchor='w').grid(row=17, column=3,sticky=W+N+E+S)
        self.location_info = Entry()
        self.location_info.grid(row=17, column=5, columnspan=4,sticky=W)

        

        # Function button for operations
        Button(self.root, height=4, text="Start inference",
            command=self.start_inference, fg='black').grid(row=19, column=3, sticky=W)
        Button(self.root, height=4, text="save in modified",
            command=self.save_modified, fg='black').grid(row=19, column=4, sticky=W)

        Button(root, height=4, text="Prev_Image", command=lambda: self.switch_image(
            'prev'), fg='blue').grid(row=19, column=5, columnspan=1, sticky=W)
        Button(root, height=4, text="Next_Image", command=lambda: self.switch_image(
            'next'), fg='blue').grid(row=19, column=6, columnspan=1, sticky=W)
    


    #load model dir
    def load_detection_model(self):
        self.detection_model_dir = filedialog.askdirectory(title=u'open detection model dir', initialdir=(
            os.path.expanduser('./Retinanet_inference_example/checkpoint/Bird_drone')))
        model_dir_info =''
        for file in os.listdir(self.detection_model_dir):
            if (file.endswith(('pt','pkl','pth'))):
                model_dir_info+=file+'\n'
        self.detection_model_info_label.config(text = model_dir_info)

    def load_classification_model(self):
        self.classification_model_dir = filedialog.askopenfilename(title=u'open classification model dir', initialdir=(
            os.path.expanduser('/home/zt253/Models/model_inference_gui/Retinanet_inference_example/checkpoint/classification')))
        model_dir_info =''+os.path.basename(self.classification_model_dir)
        self.classification_model_info_label.config(text = model_dir_info)

    
    #Generate inference info and display them
    def update_inference_info(self):
        info = dict()
        info['num_image'] = len(self.image_list)
        if (self.image_list):
            info['image_name'] = os.path.basename(self.image_list[self.image_id])
        else:
            info['image_name'] = 'None'
        info['num_detections'] = len(self.filtered_box_idx)
        info['NMS_thresh'] = self.NMS_threshold_slider.get()
        info['conf_thresh'] = self.confidence_threshold_slider.get()
        out_string = 'inference_info:\n'
        for (k, v) in info.items():
            out_string += k+':\t\t'+str(v)+'\n'
        self.inference_info_label.config(text=out_string,justify=LEFT)

    def update_NMS(self, event):
        self.NMS_threshold = self.NMS_threshold_slider.get()
        self.display_images()
        self.update_inference_info()

    def update_confidence_threshold(self, event):
        self.confidence_threshold = self.confidence_threshold_slider.get()
        self.display_images()
        self.update_inference_info()

    def open_image_dir(self):
        self.image_id = 0
        file_path = filedialog.askdirectory(title=u'open_image_dir', initialdir=(
            os.path.expanduser('/home/zt253/Models/model_inference_gui/Retinanet_inference_example/example_images/Bird_drone/60m')))
        tmp = []
        for file in os.listdir(file_path):
            if file.endswith(('.jpg','.JPG','.png')):
                print (file)
                tmp.append(os.path.join(file_path,file))
        self.image_list = sorted(tmp)
        self.out_dir = file_path+'_results'
        self.display_images()

    def open_single_image(self):
        self.image_id = 0
        file_path = filedialog.askopenfilename(title=u'open_single_image', initialdir=(
            os.path.expanduser('/home/zt253/Models/model_inference_gui/Retinanet_inference_example/example_images/Bird_drone/60m')))
        self.image_list = [file_path]
        self.out_dir = os.path.dirname(file_path)+'_results'
        os.makedirs(self.out_dir,exist_ok=True)
        self.display_images()


    def display_images(self):
        if (not self.image_list):
            return
        self.image_preview = Image.open(self.image_list[self.image_id])
        self.image_name = os.path.basename(self.image_list[self.image_id])
        detection_dir = os.path.join(self.out_dir,'detection-results',self.image_name.split('.')[0]+'.txt')
        if (os.path.exists(detection_dir)):
            with open(detection_dir,'r') as f:
                data = f.readlines()
            self.bbox = []
            for line in data:
                line = line.replace('\n','').split(',')
                self.bbox.append([float(i) for i in line[2:]]+[float(line[1]),line[0]])
            nms_idx = self.apply_NMS_threshold(self.bbox)
            conf_idx = self.apply_confidence_threshold(self.bbox)
            self.filtered_box_idx = set(nms_idx) & set(
                conf_idx)
            draw = ImageDraw.Draw(self.image_preview)
            for idx in self.filtered_box_idx:
                box = self.bbox[idx]
                draw.rectangle(
                    (box[0], box[1], box[2], box[3]), outline='red', width=8)
                font = ImageFont.load_default()
                draw.text((box[0]-10, box[1]-10),box[-1], font = font,fill =(255, 0, 0))
        self.image_preview_tk = ImageTk.PhotoImage(self.image_preview.resize(
            (self.image_preview_size[0], self.image_preview_size[1]), resample=0))
        Label(root, image=self.image_preview_tk, width=self.image_preview_size[0], height=self.image_preview_size[1]).grid(
            row=0, column=0, rowspan=20, columnspan=2, sticky=W+E+N+S)
        self.update_inference_info()

    def start_inference(self):
        #self.out_info_label.config(text = 'saved to: {}'.format(self.out_dir))
        self.altitude = int(self.altitude_entry.get())
        #self.bbox = inference(self.model_path, self.image_list, isHeight=True)

        detection_model_type = self.detection_model_type.get()
        classification_model_type = self.classification_model_type.get()
        if (detection_model_type == 'Retina-Net'):
            model_dir = os.path.join(self.detection_model_dir,'final_model.pkl')
            image_out_dir = os.path.join(self.out_dir,'visualize-results')
            text_out_dir = os.path.join(self.out_dir,'detection-results')
            csv_out_dir = os.path.join(self.out_dir,'detection_summary.csv')
            os.makedirs(image_out_dir,exist_ok=True)
            os.makedirs(text_out_dir,exist_ok=True)
            altitude_list = [self.altitude for _ in self.image_list]
            date_list = [self.date_info.get() for _ in self.image_list]
            location_list = [self.location_info.get() for _ in self.image_list]
            inference_mega_image_Retinanet(image_list = self.image_list,
                model_dir = model_dir,
                image_out_dir = image_out_dir,text_out_dir = text_out_dir,csv_out_dir = csv_out_dir,
                scaleByAltitude = True,defaultAltitude = altitude_list,
                date_list = date_list,location_list = location_list,
                visualize = True,device = torch.device('cpu'),model_type = 'Bird_drone')
        
        self.display_images()

    def apply_NMS_threshold(self,bbox):
       return py_cpu_nms(bbox, self.NMS_threshold/100.0)

    def apply_confidence_threshold(self,bbox):
        return [i for i, j in enumerate(
            bbox) if (j[4] >= self.confidence_threshold/100.0)]

    def save_modified(self):
        image = self.image_preview
        image_out_dir = os.path.join(self.out_dir,'visualize-results_modified')
        text_out_dir = os.path.join(self.out_dir,'detection-results_modified')
        os.makedirs(image_out_dir,exist_ok=True)
        os.makedirs(text_out_dir,exist_ok=True)
        image.save(os.path.join(image_out_dir,self.image_name))
        print (image_out_dir,text_out_dir)
        with open(os.path.join(text_out_dir,self.image_name.split('.')[0]+'.txt'),'w') as f:
            for idx in self.filtered_box_idx:
                box = self.bbox[idx]
                f.writelines('{},{},{},{},{},{}\n'.format(box[5],box[4],box[0],box[1],box[2],box[3]))
    def switch_image(self, direction='next'):
        if (direction == 'next'):
            self.image_id = min(len(self.image_list)-1, self.image_id+1)
        else:
            self.image_id = max(0, self.image_id-1)
        self.update_inference_info()
        self.display_images()


def about():
    print('open')


if __name__ == '__main__':
    root = Tk()
    root.title('model_inference_GUI')
    root.geometry('400x200')
    ClassifyGUI(config_data=data, root=root)
    root.mainloop()
