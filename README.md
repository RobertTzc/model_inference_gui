# Waterfowl Detection and Classification Inference Interface

## Purpose
The Waterfowl Detection and Classification Inference Interface is built to allow users to run detection and classification algorithms on images of waterfowl taken with a drone for the purpose of population monitoring. Key features of the interface include the option to load individual images or a folder of images, selection of different detection and classification models, assigning image height to improve model inference, choosing confidence scores of interest, inputting location and date information for use in future analyses of the users' choice, and viewing detection results with different confidence score selections.

## Software/Hardware Requirements
Anaconda: https://www.anaconda.com/ (will need system admin privileges)

Python: https://www.python.org/ (will need system admin privileges)

Windows 10
- Anaconda3
- Python 3.9

Other

# Installation and Set Up

## Opening Anaconda
To start Anaconda on Windows, you may either:
1. Use the Start Menu to Search for Anaconda Prompt and select Anaconda Prompt to open the command window for Anaconda
2. Navigate to the Anaconda Prompt icon [default location C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Anaconda3 (64-bit)] (or Anaconda Prompt shortcut icon) and double-click to run the command window for Anaconda

## Cloning the Repository from GitHub
To clone the repository from GitHub there are a two main options:
1. Download and unzip directly from GitHub (recommended)
* Navigate to the GitHub repository for the model_inference_gui here: https://github.com/RobertTzc/model_inference_gui/
* In the upper right, click on the <>Code dropdown arrow and select Download ZIP
* After download, extract the model_inference_gui.zip folder and move it to the desired location on your machine. Recommended is under C:\Users\\*yourusername*
* If you are successful, skip to Installation of Required Packages.

2. Use the Anaconda Prompt cmd window to clone the repository
* In Anaconda Prompt, run:
```
conda install -c anaconda git
```     
* This may require admin privileges. If the previous line successfully ran, next run:
```
git clone https://github.com/RobertTzc/model_inference_gui.git
```
* If the previous lines work, skip to Installation of Required Packages.
* If you do not have admin privileges, and the previous lines did not run, install git by downloading the program here: https://git-scm.com/downloads
* Open Git Bash (Start Menu -> Search for Git Bash)
* In Git Bash, run the following line of code:
```
git clone https://github.com/RobertTzc/model_inference_gui.git
```

If needed, for additional help with this step, please refer to the link here: https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository

## Model download:
* After cloning the repository onto your device, download the checkpoint_gui.zip folder here: https://drive.google.com/file/d/1JervM9OLvnueS_2aiufIteBUeNcEVjob/view?usp=sharing

* After download, extract the checkpoint_gui.zip folder (commonly in the "Downloads" folder) and place the "classifier" and "Retinanet" folders in the checkpoint folder under the model_inference_gui folder.

* Using the steps under #1 of cloning the repository above and the model download steps, the "classifier" and "Retinanet" folders should be located at: C:\Users\\*yourusername*\model_inference_gui\checkpoint

## Creation and Activation of Virtual Environment
Create your virtual environment in Anaconda Prompt. The following is a demo using Python 3.8 and Anaconda to create a virtual environment: *Note*: You may name the virtual environment whatever you choose. In our example, we use the name "torch_py3".

* Open Anaconda Prompt 

* Creating the virtual environment:
```
conda create -n torch_py3 python==3.8
```
* Activating the virtual environment:
```
conda activate torch_py3
```
For more information on creating and activating virtual environments, see: https://conda.io/projects/conda/en/latest/user-guide/getting-started.html 

## Installation of Required Packages
### 1. Within a Virtual Environment
After creating and activating a virtual environment, we recommend installing packages within the virtual environment.

* To install the required packages, run the following code after creating and activating the virtual environment:
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install -c anaconda tk
pip install opencv-contrib-python
pip install Pillow==6.2.2
pip install pandas
pip install pyexiv2 #For mac os please use pip install pyexiv2==2.3.1
pip install matplotlib
pip install -i https://test.pypi.org/simple/ WaterFowlTools
pip install packaging
pip install kiwisolver
pip install cycler
pip install tqdm

```
*Note*: For installation of Pillow==6.2.2, 6.1 also works. Try using Pillow==6.1 if errors arise attempting to install Pillow==6.2.2

### 2. Direct installation - Not Recommended
Packages can also be installed directly with pip (not recommended) if desired. If you installed packages in the virtual environment under step 1 above, you skip this part.

* For direct installation, run the following code:
```
For windows:
  pip3 install torch torchvision torchaudio
For Linux:
  pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
The rest are identical on both systems:

pip install opencv-contrib-python
pip install Pillow==6.1 #6.2 also work
pip install pandas
pip install pyexiv2
pip install matplotlib
pip install -i https://test.pypi.org/simple/ WaterFowlTools
pip install packaging
pip install kiwisolver
pip install cycler
pip install tqdm

```

## Future Activation of the Virtual Environment

After creation of the Virtual Environment the first time, you do NOT need to re-create the virtual environment and re-install the packages each time you use Anaconda and wish to run the Waterfowl Detection and Classification Inference Interface.

* To activate the Virtual Environment in future sessions, simply run the following code after opening Anaconda Prompt.
```
conda activate torch_py3
```
*Note*: Keep in mind that the name of your virtual environment may be different than ours (user's choice). Our virtual environment is named: torch_py3

# GUI Usage
After cloning the repository, downloading the model checkpoint folders, creating the virtual environment, and installing the required packages, you are ready to run the Waterfowl Detection and Classification Interface. 

*Note*: Once everything up to this point has been done once successfully, you will only need to run the following lines of code to activate and use the GUI in the future.

## Running the Interface
* Ensure you are in your virtual environment in Anaconda that you previously created for running the GUI in the above steps. In Anaconda prompt, you should see (*yourvirtualenvironment*) *YourDirectory*>

* If you see (base) *YourDirectory*>   , activate your virtual environment by running:
```
conda activate torch_py3
```
*Note*: Keep in mind that the name of your virtual environment may be different than ours (user's choice). Our virtual environment is named: torch_py3

* Once you are in your virtual environment, change your directory in Anaconda to the model_inference_gui folder using the code: cd C:\Users\\*yourusername*\model_inference_gui
```
cd C:\Users\yourusername\model_inference_gui
```
* Next, run the GUI_ver2.py file to start the interface program:
```
python GUI_ver2.py
```
*Note*: If you receive an error referring to a missing module, ensure you are in the virtual environment created for running this interface and all packages were successfully installed in the above steps.

*Note*: If you receive an error "No file found GUI_ver2.py", ensure that you are in the correct directory and are under the model_inference_gui folder in Anaconda prompt.

The basic user interface looks like the image below this with the major sections labeled for navigation. The following sections will explain how to use the interface.
![Screenshot from 2023-01-20 11-59-37](https://user-images.githubusercontent.com/71574752/213773521-64013052-a17f-430c-9e7e-ac509a85f9ac.png)

## Loading Images 
We provide two differnet options for loading and running images through the detectors and classifiers, a single image (open_single_image) or a folder of images (open_image_dir) option.

To load images, click **File** in the menu bar (top left of the interface) and you will be prompted with options: 
* open_single_image: 
  * will let the user select one image for running detection and classification models
* open_image_dir:
  * will let the user specify a folder of images for running detection and classification models
  * all the files of type 'jpg','JPG', and 'png' will be automatically loaded into the interface to run on all files at once

## Selecting Models and Loading Model Weights
We provide multiple detection and classification models for the user to select which model to use for inference on the images. See below for expected model performance results, and choose the model most appropriate for your scenario.

*Note*: The classification models are built on images collected at 15 meters (0.38 cm/pixel GSD). It is not recommended to run images from higher altitudes or with lower resolution as you may get unexpected results and poor classification performance. If a classification model is not desired, select "Disable" under Classification Model Selection.

***Insert Model Test Results Here***

* Select the desired detection and classification model by clicking the empty circle to the left of the name of the desired model in the upper-right portion of the GUI.

After selecting the models to use, the user must specify the folder containing the model weights.

### Loading Detection Model Weights
1. Select the **Load Model** button to load corresponding model weights. 
2. In the pop-up window, navigate to the checkpoint folder (model_inference_gui\checkpoint).
3. Select the desired model folder 
   * For Retina-Net or Retina-Net_KNN, open the Retinanet folder and select Bird_Drone or Bird_Drone_KNN and choose select folder.
   * For YOLOv5, select the Yolov5 folder and choose select folder
   
Once selected the GUI will automatically show the model weights file(s) found in the folder. 

### Loading Classification Model Weights
If a classification model is not selected, skip this section of loading the classification model weights.
1. Select the **Load Model** button to load corresponding model weights. 
2. In the pop-up window, navigate to the checkpoint folder (model_inference_gui\checkpoint).
3. Open the "classifier" folder
4. Open the folder that corresponds to the chosen classifier
   * For Res18, open the Res18_Bird_I folder
5. Select the "model.pth" file and select open to load the model weights.
   
Once selected the GUI will automatically show the model weights file. 

## Altitude Input
* Some of the models rescale the images based on altitude (or ground sample distance), so the user must input the altitude (in meters) the images were taken at. The altitude must be an integer and should represent meters in altitude.

Heights at which our models were trained and corresponding GSDs:
```
Altitude       GSD
(meters)  (cm/pixel)
   15         0.342
   30         0.684
   60         1.368
   90         2.052
```

## Threshold Adjustment Bars

The GUI provides the user with two sliding bars for adjusting the confidence and NMS thresholds. 

***Confidence Threshold:***

***NMS Threshold:***

***Recommended Scenarios and Thresholds:***

* To select the desired thresholds, click and drage the icons to select the appropriate value for the threshold.

## Optional Image Info

The user has the option to input the Location and Date of the imagery collected. Providing this information is optional, and, if provided, it is added to the final result .csv file.

# Starting inference

* After the above steps are complete, click **Start inference**, and the model will start to detect the birds. It may take up to 3 minutes per image depending on the device used and models selected. 

***List Tested Machines, Altitudes, and time for Inference Here***

Once the inference on all images is complete, the visual detection results will be shown on the image preview screen:

![Screenshot from 2023-01-20 11-58-37](https://user-images.githubusercontent.com/71574752/213773610-0a97c2f4-8f3a-4743-ba2a-c0f59b29050f.png)

At this time, the user may view the detection results for the input thresholds and models selected. Thresholds can be modified and applied anytime upon completion of inference (i.e. you have detection results). After running the inference, the photo(s) will show the detection results for the selected thresholds. The user may adjust these thresholds and the display will change according to the thresholds to show what results the new thresholds will generate. If the user finds better results at a different threshold than originally ran, and wants to save these results, the user may save the modified results by clicking **save in modified** to save the adjusted result without overwriting the original result.

# Saving Results
By default, a folder of the results will be created where the input image or image folder is located. The folder will have the name of *inputimage*_result or *inputfolder*_result and will be located adjacent to the input image or folder. 

If the user modifies the thresholds as described above and chooses to save the results using **save in modified**, it will be saved adjacent to the input and results folder under *inputimage*_modified or *inputfolder*_modified.

Example: 

If the input folder *15m* is located here: C:\Users\*yourusername*\model_inference_gui\example_images\Bird_drone\, 

the results folder *15m_result* will be located in the same place, here: C:\Users\*yourusername*\model_inference_gui\example_images\Bird_drone\, 

and the modified results *15m_modified* will be located in the same place, here: C:\Users\*yourusername*\model_inference_gui\example_images\Bird_drone\

***Add Example Photo of File Explorer?***

The saved outputs and output structure are shown below:
```
Input_image_folder: foldername 
├── image_name1.jpg
├── image_name2.jpg
├── image_name3.jpg
└── ...
Result_image_folder: foldername_result
├── detection_summary.csv - csv file of detection and classification results
│   ├── image_name
│   ├── date (if user provided)
│   ├── location (if user provided)
│   ├── altitude (user provided)
│   ├── latitude_meta (if located in Image Metadata)
│   ├── longitude_meta (if located in Image Metadata)
│   ├── altitude_meta (if located in Image Metadata)
│   ├── num_birds - Number of Birds Detected in the image
│   ├── time_spent(sec) - Number of Seconds it took for Detection on the image
├── detection-results (folder) - text files of the bird detections
│   ├── image_name1.txt
│   ├── image_name2.txt
│   ├── image_name3.txt
│   └── ...
├── visual-results (folder) - images showing the bird detection (and classification, if applicable)
│   ├── image_name1.jpg
│   ├── image_name2.jpg
│   ├── image_name3.jpg
│   └── ...
├── *classification-results (folder, optional) - text files of the bird classifications
│   ├── image_name1.txt
│   ├── image_name2.txt
│   ├── image_name3.txt
│   └── ...
├── *detection-results_modified (folder, optional) - text files of the bird detections if thresholds modified
│   ├── image_name1.txt
│   ├── image_name2.txt
│   ├── image_name3.txt
│   └── ...
├── *visual-results_modified (folder, optional) - images showing the bird detection (and classification, if applicable) if thresholds modified
│   ├── image_name1.jpg
│   ├── image_name2.jpg
│   ├── image_name3.jpg
│   └── ...
```
