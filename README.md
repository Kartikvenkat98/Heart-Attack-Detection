# Heart-Attack-Detection

## Author
**Kartik Venkataraman**


## Preprocessing

(1) mkdir dataset && mkdir videos && mkdir model && mkdir frames && mkdir fg-extract <br>
(2) Download dataset from [here](https://drive.google.com/drive/folders/16HhfMovQMS8iMgBVFRwJ4tmpOwmVzprG?usp=sharing) by logging with TAMU email and put them under ``dataset`` folder <br>
(3) Download videos from [here](https://drive.google.com/drive/folders/1ka5vGFS09oeEejoiIFoFbgdrHPmMXCxr?usp=sharing) by logging with TAMU email and put them under ``videos`` folder <br>


## Training the model

``python train.py`` <br>
This command will save a model a model.h5 under ``model`` folder. <br>

Alternatively, you can directly use my uploaded model.


## Testing the model

``python test.py`` <br>
This command will predict infraction in the static images in the ``test`` folder in ``dataset`` directory. <br>
This shows 91.75% accuracy and 92.85% sensitivity.


## Detection in Videos

There are three steps for detecting possible heart-attacks from videos.

### (1) Frame Generation from videos

``python frame_generator.py`` <br>
This command will generate frames from a video in the ``videos`` folder and saving it in the ``frames`` directory. <br>
For now, we have to manually give the video filename inside the code but will be
changed to taking argument from the terminal. <br>


### (2) Instance Segmentation and Background Removal

``python seg_backrem.py`` <br>
The next step is the instance segmentation and background removal from each of the extracted frames in the ``frames`` directory. <br>
The background is changed to magenta color for having the contrast for the segmented image. <br>
This command will generate the segmented image for the frames and save them in ``fg-extract`` folder. <br>


### (3) Evaluation

``python test_modified.py`` <br>
The final step is the evaluation of each of the segmented frames in the ``fg-extract`` directory. <br>
This will also generate the required plot that at a specific time,
what is the predicted probability of heart attack and also creates the json file. <br>

