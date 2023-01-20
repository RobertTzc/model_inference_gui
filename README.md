# SpeciesLabelTool


## Installation

### Clone the repository
You can either use the cmd window to clone the repo using the following command or you can refer to the link here:https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository
```
git clone https://github.com/RobertTzc/model_inference_gui.git
```
We provided two different ways of installing the packages, through virtual env and direct installation
### Virtual env

Virtual env is recommended to be used in order to install the package, here Anaconda is recommended to be used, link to the Anaconda https://www.anaconda.com/, once you have installed the Anaconda , refer here to create you virtual env https://conda.io/projects/conda/en/latest/user-guide/getting-started.html. It is recommend to create the env along with python 3.8, demo cmd is here:
```
conda create -n torch_py3 python==3.8
```
once env is created use, eg:
```
conda activate torch_py3
```
once you have activated the env, make sure you are under the env you created and run the following commands:
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install -c anaconda tk
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
pip install efficientnet_pytorch
```
### Direct installation
Packages can also be installed with direct pip :

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
Please download the model here https://drive.google.com/file/d/1nyAW3DRDkTjqvoENW3ysyK9TF6zEHpKx/view?usp=sharing and put the extracted **checkpoint** folder under folder Retinanet_inference_example folder. Current checkpoint folder contains Retinanet Model weights and classification model weights.

## GUI usage
Once the installation is complete and model weights is downloaded and placed, run the GUI_ver2.py file to start the program, you can use the following command to start the GUI:
```
//cd to the current directory
python GUI_ver2.py
```
![Screenshot from 2023-01-20 11-59-37](https://user-images.githubusercontent.com/71574752/213773521-64013052-a17f-430c-9e7e-ac509a85f9ac.png)

In the image above shows a default UI interface of the Inference tool where major components are labeled. In the next section will describe how to use the UI.

### Loading images 
We provide two differnet ways of loading images: single image loading and image folder loading. They are located in the  menu ba, click **File** and you shall be prompt with options 
```
open_single_image: 
    will let user select exact one image from local machine
open_image_dir:
    will let user specify where the image folder locates, and will automatically load all the files end with 'jpg','JPG','png'
```

## Specifying model types and load model weights
Both detection model and classification model provides several options for the user to select what type of model to be applied, by default Retinanet is selected and no classification is selected.

Once user decided which model to use, user can specify the location of the model weight by click button **Load Model** to load corresponding model weights. Again only the folder directory is needed, the GUI will automatically shows the model weights founded under the folder. If classification model type is not selected, the Load Model for the classification model can be skipped.

## Altitude input
Some of the models requires altitude to help with the image rescale process, user need to input the current images altitude before the inference start, the altitude has to be an integer.

## Threshold adjust bar

The GUI provided two sliding bar for adjusting the confidence and NMS threshold, they can be applied anytime if there is detection results showing on the preview, when user adjust these threshold and found a better result, user can save the modified result by clicking **save in modified** to save the adjusted result as extra result.

## Image info input

The GUI also collect two extra information, which are location and date of the collected image, these information are optional and will only be used to add to the final result csv file.

## Start inference

After all the step above has been done, click start inference and the model will start to detect the birds, it may take up to 1 minutes per image depend on the device, once the **start inference** button transfer from grey to active again, the inference is complete, and results will be shown on image preview screen. See figure below:

![Screenshot from 2023-01-20 11-58-37](https://user-images.githubusercontent.com/71574752/213773610-0a97c2f4-8f3a-4743-ba2a-c0f59b29050f.png)

## Saving Results
By default, this tool will create an folder where the input image folder located, with an extension of '_result', if a better threshold is being found by user and saved, it will be saved with extension **'_modified'**

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
├── *visual-results_modified
│   ├── image_name1.jpg
│   ├── image_name2.jpg
│   ├── image_name3.jpg
│   └── ...
```
