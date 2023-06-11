from retinanet_inference import inference
from retinanet_inference import py_cpu_nms
from  create_category_list_json import *
from tkinter import filedialog, dialog
from tkinter import *  
import os
import os.path as path
import glob
from PIL import Image, ImageTk
from PIL import Image, ImageDraw
import collections
class ClassifyGUI():
	def __init__(self,config_data,root): 
		self.root = root
		self.config = config_data
		self.root.geometry(str(self.config['GUIResolution'][0])+'x'+str(self.config['GUIResolution'][1]))
		self.large_view_size = [int(self.config['RelativeLayoutImageView'][0]*self.config['GUIResolution'][0]),int(self.config['RelativeLayoutImageView'][1]*self.config['GUIResolution'][1])]
		self.image_list = []
		self.image_id = 0
		self.model_dir = ''
		self.detection_boxes = dict()
		self.confidence_threshold  = 50
		self.NMS_threshold = 50
		self.isInferencedd = False
		self.bbox_idx_confidence = []
		self.bbox_idx_NMS = []
		self.config_UI()
		
	def config_UI(self):
		menubar = Menu(self.root) 
		filemenu = Menu(menubar,tearoff=0)  
		filemenu.add_command(label="open_single_image",command = self.open_single_image)
		filemenu.add_command(label = "open_image_dir",command = self.open_image_dir)
		filemenu.add_command(label = "Load_models",command = self.Load_models)
		filemenu.add_separator()  
		filemenu.add_command(label="Exit", command=root.quit)  
		menubar.add_cascade(label="File", menu=filemenu)  
		helpmenu = Menu(menubar, tearoff=0)  
		helpmenu.add_command(label="About", command=about)  
		menubar.add_cascade(label="Help", menu=helpmenu)    
		self.root.config(menu=menubar)
		self.LargeImage = Image.new('RGB', (self.large_view_size[0],self.large_view_size[1]))
		self.largeImage_tk = ImageTk.PhotoImage(self.LargeImage)
		self.large_view_window = Label(root, image=self.largeImage_tk,width=self.large_view_size[0],height =self.large_view_size[1])
		self.large_view_window .grid(row=0, column=0,rowspan = 20,columnspan=2,sticky=W+E+N+S)

		self.model_dir_label = Label(root,text = 'model_name: '+ 'retinanet',anchor='w')
		self.model_dir_label.grid(row = 1,column = 3,columnspan = 2,sticky = W)
		self.inference_info_label = Label(root,text = 'inference information',anchor='w')

		self.inference_info_label.grid(row = 2,column = 3,columnspan = 2,sticky = W)
		Label(root, text = "confidence_threshold(%)").grid(row = 10,column = 3,columnspan = 2)
		self.confidence_threshold_slider = Scale(root, from_=10, to=100, orient=HORIZONTAL,length=400)
		self.confidence_threshold_slider.bind("<ButtonRelease-1>", self.update_confidence_threshold)
		self.confidence_threshold_slider.grid(row = 11, column = 3,columnspan=4)
		self.confidence_threshold_slider.set(self.confidence_threshold)
		Label(root, text = "NMS_threshold(%)").grid(row = 12,column = 3,columnspan = 2)
		self.NMS_threshold_slider = Scale(root, from_=10, to=100, orient=HORIZONTAL,length=400)
		self.NMS_threshold_slider.bind("<ButtonRelease-1>", self.update_NMS)
		self.NMS_threshold_slider.grid(row = 13, column = 3,columnspan=4)
		self.NMS_threshold_slider.set(self.NMS_threshold)
		Button(root,width=25,height=4,text = "Start inference",command = self.start_inference,fg='black').grid(row = 16, column = 3,columnspan=1)
		#Button(root,width=25,height=4,text = "Update result",command = self.update_result,fg='black').grid(row = 16, column = 4,columnspan=1)
		Button(root,width=25,height=4,text = "save detection in txt",command = self.save_detection_txt,fg='black').grid(row = 17, column = 3,columnspan=1)
		Button(root,width=25,height=4,text = "save detection in image",command = self.save_detection_image,fg='black').grid(row = 17, column = 4,columnspan=1)
		
		Button(root,width=25,height=4,text = "Next_Image",command = lambda: self.switch_image('next'),fg='blue').grid(row = 19, column = 4,columnspan=1)
		Button(root,width=25,height=4,text = "Prev_Image",command = lambda: self.switch_image('prev'),fg='blue').grid(row = 19, column = 3,columnspan=1)
	def update_inference_info(self):
		info = dict()
		info['num_image'] = len(self.image_list)
		if (self.image_list):
			info['image_name'] = self.image_list[self.image_id].split('/')[-1]
		else:
			info['image_name'] = 'None'
		info['num_detections'] = len(set(self.bbox_idx_NMS)&set(self.bbox_idx_confidence))
		info['NMS'] = self.NMS_threshold_slider.get()
		info['confidence_threshold'] = self.confidence_threshold_slider.get()
		out_string = 'inference_info:\n'
		for (k,v) in info.items():
			out_string+=k+':\t'+str(v)+'\n'
		self.inference_info_label.config(text = out_string)
	def update_NMS(self,event):
		self.NMS_threshold = self.NMS_threshold_slider.get()
		self.display_images()
		self.update_inference_info()
	def update_confidence_threshold(self,event):
		self.confidence_threshold = self.confidence_threshold_slider.get()
		self.display_images()
		self.update_inference_info()
	def open_image_dir(self):
		self.image_id = 0
		file_path = filedialog.askdirectory(title=u'open_image_dir', initialdir=(os.path.expanduser('/home/zt253/data/UnionData/2019_Summer_decoy')))
		self.image_list = sorted(glob.glob(file_path+'/*.JPG'))+sorted(glob.glob(file_path+'/*.jpg'))
		self.isInferenced = False
		self.display_images()
	def open_single_image(self):
		self.image_id = 0
		file_path = filedialog.askopenfilename(title=u'open_single_image', initialdir=(os.path.expanduser('/home/zt253/data/UnionData/2019_Summer_decoy')))
		self.image_list = [file_path]
		self.isInferenced = False
		self.display_images()
	def Load_models(self):
		file_path = filedialog.askopenfilename(title=u'Load the models', initialdir=(os.path.expanduser('/home/zt253/Models/retinanet_UnionData/checkpoint/Eagle_bluff_sep29_PNratio0.2')))
		self.model_path = file_path
		self.isInferenced = False
		#self.model_dir_label.config(text = 'model_dir: '+self.model_path)
	def display_images(self):
		if (self.image_list):
			self.LargeImage = Image.open(self.image_list[self.image_id])
			if (self.isInferenced):
				print ('draw box')
				self.apply_NMS_threshold()
				self.apply_confidence_threshold()
				common_idx = set(self.bbox_idx_NMS) & set(self.bbox_idx_confidence)
				#print ('common idx',common_idx,self.bbox_idx_NMS,self.bbox_idx_confidence)
				draw = ImageDraw.Draw(self.LargeImage)
				current_bbox = self.bbox[self.image_list[self.image_id].split('/')[-1]]
				for idx in common_idx:
					box = current_bbox[idx]
					draw.rectangle((box[1],box[2],box[3],box[4]),outline='red',width = 8)
			self.largeImage_tk = ImageTk.PhotoImage(self.LargeImage.resize((self.large_view_size[0],self.large_view_size[1]),resample=0))
			Label(root, image=self.largeImage_tk,width=self.large_view_size[0],height =self.large_view_size[1]).grid(row=0, column=0,rowspan = 20,columnspan=2,sticky=W+E+N+S)
		
	def start_inference(self):
		self.bbox = inference(self.model_path,self.image_list,isHeight = True)
		#print (self.bbox)
		self.isInferenced = True
		self.display_images()
	def apply_NMS_threshold(self):
		current_bbox = self.bbox[self.image_list[self.image_id].split('/')[-1]]
		self.bbox_idx_NMS = py_cpu_nms(current_bbox,self.NMS_threshold/100.0)

	def apply_confidence_threshold(self):
		current_bbox = self.bbox[self.image_list[self.image_id].split('/')[-1]]
		self.bbox_idx_confidence = [i for i,j in enumerate(current_bbox) if (j[0]>=self.confidence_threshold/100.0)]
		
	def save_detection_txt(self):
		root_path = os.path.dirname(self.image_list[0])
		saved_path = root_path+'/'+'detection'
		if (not os.path.isdir(saved_path)):
			os.mkdir(saved_path)
		image_name = self.image_list[self.image_id].split('/')[-1]
		ext = image_name.split('.')[-1]
		with open(saved_path+'/'+image_name.replace('.'+ext,'.txt'),'w') as f:
			for box in self.bbox[image_name]:
				f.writelines('{} {} {} {} {}\n'.format(box[0],box[1],box[2],box[3],box[4]))
	def save_detection_image(self):
		im = Image.open(self.image_list[self.image_id])
		root_path = os.path.dirname(self.image_list[0])
		saved_path = root_path+'/'+'detection'
		image_name = self.image_list[self.image_id].split('/')[-1]
		if (not os.path.isdir(saved_path)):
			os.mkdir(saved_path)
		draw = ImageDraw.Draw(im)
		current_bbox = self.bbox[self.image_list[self.image_id].split('/')[-1]]
		for idx in set(self.bbox_idx_NMS)&set(self.bbox_idx_confidence):
			box = current_bbox[idx]
			draw.text(xy=(box[1]+15,box[2]+15), text=str(int(100*box[0])), fill=(0, 255, 255), )
			draw.rectangle((box[1],box[2],box[3],box[4]),outline='red',width = 8)
		im.save(saved_path+'/'+image_name)
		print (saved_path+'/'+image_name)
	def save_anno(self,label):
		if(not os.path.isdir(os.path.split(self.result_file)[0])):
			os.mkdir(os.path.split(self.result_file)[0])
		current_box = self.bbox[self.bird_id]
		bird_exist = False
		if(path.exists(self.result_file)):
			with open(self.result_file, "r") as f1,open("%s.bak" % self.result_file, "w") as f2:
				for line in f1.readlines():
					box = [int(i) for i in line.split(',')[2:]]
					if (collections.Counter(current_box) == collections.Counter(box)):
						bird_exist = True
						f2.writelines('bird,{},{},{},{},{}\n'.format(label,current_box[0],current_box[1],current_box[2],current_box[3]))
					else:
						f2.writelines(line)
				if (bird_exist == False):
					f2.writelines('bird,{},{},{},{},{}\n'.format(label,current_box[0],current_box[1],current_box[2],current_box[3]))
			os.remove(self.result_file)
			os.rename("%s.bak" % self.result_file, self.result_file)
		else:
			with open(self.result_file, "w") as f:
				f.writelines('bird,{},{},{},{},{}\n'.format(label,current_box[0],current_box[1],current_box[2],current_box[3]))

	def switch_image(self,direction = 'next'):
		if (direction=='next'):
			self.image_id= min(len(self.image_list)-1,self.image_id+1)
		else:
			self.image_id=max(0,self.image_id-1)
		self.update_inference_info()
		self.display_images()

def about():
	print ('open')


if __name__ == '__main__':
	root = Tk()
	root.title('model_inference_GUI')
	root.geometry('400x200')
	ClassifyGUI(config_data=data,root = root)
	root.mainloop()
