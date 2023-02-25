# Waterfowl Detection and Classification Inference Interface


## Installation

### Clone the Repository
You can either use the cmd window to clone the repo using the following command or you can refer to the link here: https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository
```
git clone https://github.com/RobertTzc/model_inference_gui.git
```
### Installation of Required Packages
#### 1. Virtual Environment

It is recommended to use the virtual environment to install the packages through Anaconda (https://www.anaconda.com/). After installation of Anaconda, create your virtual environment (https://conda.io/projects/conda/en/latest/user-guide/getting-started.html). Here is a demo using Python 3.8 and Anaconda to create a virtual environment:
```
conda create -n torch_py3 python==3.8
```
Activating the virtual environment:
```
conda activate torch_py3
```
Installation of required packages once you have activated the environment:
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install -c anaconda tk
pip install resnet_pytorch
pip install opencv-contrib-python
pip install Pillow==6.2.2 #6.1 also works
pip install pandas
pip install pyexiv2
pip install matplotlib
pip install -i https://test.pypi.org/simple/ WaterFowlTools
pip install packaging
pip install kiwisolver
pip install cycler
pip install efficientnet_pytorch
```
#### 2. Direct installation
Packages can also be installed directly with pip (not recommended):

```
For windows:
  pip3 install torch torchvision torchaudio
For Linux:
  pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
The rest are identical on both systems:
pip install efficientnet_pytorch
pip install resnet_pytorch
pip install opencv-contrib-python
pip install Pillow==6.1 #6.2 also work
pip install pandas
pip install pyexiv2
pip install matplotlib
pip install -i https://test.pypi.org/simple/ WaterFowlTools
pip install packaging
pip install kiwisolver
pip install cycler
```

## Model download:
Please download the model checkpoint folder here: https://drive.google.com/file/d/1nyAW3DRDkTjqvoENW3ysyK9TF6zEHpKx/view?usp=sharing 
Extract the **checkpoint** folder and place under the Retinanet_inference_example folder. The checkpoint folder currently contains Retinanet Model weights and classification model weights.

## GUI Usage
After installation and placement of the checkpoint folder, run the GUI_ver2.py file to start the program:
```
//cd to the current directory
python GUI_ver2.py
```
![Screenshot from 2023-01-20 11-59-37](https://user-images.githubusercontent.com/71574752/213773521-64013052-a17f-430c-9e7e-ac509a85f9ac.png)

The image above shows a default user interface of the inference tool with major components labeled. The following sections will explain how to use the UI.

### Loading Images 
We provide two differnet options for loading and running images through the detectors and classifiers: single images and batch option (folders). 
To load images, click **File** in the menu bar and you will be prompted with options: 
```
open_single_image: 
    will let user select exact one image from local machine
open_image_dir:
    will let user specify where the image folder is located, and will automatically load all the files end with 'jpg','JPG','png' to run at once
```

### Specifying Model Types and Loading Model Weights
We provide multiple options of detection and classification models for the user to select what type of model to use for inference on the images. The default is the Retinanet detection model with no classification model selected.

After selecting the models to use, the user must specify the folder containing the model weight using the **Load Model** button to load corresponding model weights. Again only the folder directory is needed, once selected the GUI will automatically show the model weights founded under the folder. If a classification model is not selected, loading the classification model weights should be skipped.

### Classification model

Classification model can be activated if needed, the classification is most suitable with image at altitude of 15meters, higher altitude images may having unexpected effects.

### Altitude input
Some of the models rescale the images based on altitude (or ground sample distance), so the user must input the altitude (in meters) the images were taken at. The altitude must be an integer and should represent meters in altitude.
Heights at which our models were trained and corresponding GSDs:
```
Altitude       GSD
(meters)  (cm/pixel)
   15         0.342
   30         0.684
   60         1.368
   90         2.052
```

### Threshold Adjustment Bars

The GUI provides the user with two sliding bars for adjusting the confidence and NMS thresholds. They can be modified and applied anytime if there is detection results showing on the photo preview to adjust model performance. After the user adjusts these thresholds and finds better results, the user may save the modified results by clicking **save in modified** to save the adjusted result as an extra result.

### Image Info

The GUI also prompts the user for other information, the location and date of the collected image(s). Providing this information is optional and if provided is added to the final result csv file.

## Starting inference

After all of the above steps are complete, click **start inference** and the model will start to detect the birds. It may take up to 1 minute per image depending on the device used and models selected. Once the **start inference** button transfers from grey to active again, the inference is complete, and results will be shown on image preview screen:

![Screenshot from 2023-01-20 11-58-37](https://user-images.githubusercontent.com/71574752/213773610-0a97c2f4-8f3a-4743-ba2a-c0f59b29050f.png)

## Saving Results
By default, this tool will create a folder where the input image or image folder is located, with an extension of '_result'. After inference, if the user modifies the thresholds as described above and saved by the user, it will be saved with extension **'_modified'**.

The GUI outputs and output structure are shown below:
```
Input_image_folder: xxx 
├── image_name1.jpg
├── image_name2.jpg
├── image_name3.jpg
└── ...
Result_image_folder: xxx_result
├── detection-results
│   ├── image_name1.txt
│   ├── image_name2.txt
│   ├── image_name3.txt
│   └── ...
├── visual-results
│   ├── image_name1.jpg
│   ├── image_name2.jpg
│   ├── image_name3.jpg
│   └── ...
├── detection_summary.csv
└── *detection-results_modified
│   ├── image_name1.txt
│   ├── image_name2.txt
│   ├── image_name3.txt
│   └── ...
└── *classification-results
│   ├── image_name1.txt
│   ├── image_name2.txt
│   ├── image_name3.txt
│   └── ...
├── *visual-results_modified
│   ├── image_name1.jpg
│   ├── image_name2.jpg
│   ├── image_name3.jpg
│   └── ...
```
