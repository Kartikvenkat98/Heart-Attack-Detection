# Heart-Attack-Detection

## Author
**Kartik Venkataraman**


## Preprocessing

(1) mkdir dataset && mkdir videos<br>
(2) Download dataset from [here](https://drive.google.com/drive/folders/16HhfMovQMS8iMgBVFRwJ4tmpOwmVzprG?usp=sharing) by logging with TAMU email and put them under ``dataset`` folder <br>
(3) Download videos from [here](https://drive.google.com/drive/folders/1ka5vGFS09oeEejoiIFoFbgdrHPmMXCxr?usp=sharing) by logging with TAMU email and put them under ``videos`` folder <br>


## Training the model

``python train.py`` <br>
This command will train the CNN on the images in the ``train`` folder inside the ``dataset`` directory and save the trained model as model.h5 under ``model`` folder. <br>

Alternatively, you can directly use the trained uploaded model.


## Testing the model

``python test.py`` <br>
This command will predict infraction on the static images in the ``test`` folder inside the ``dataset`` directory. <br>
This shows 91.75% accuracy and 92.85% sensitivity.


## Detection in Videos

There are three steps for detecting possible heart-attacks from videos.

### (1) Frame Generation from videos

``python frame_generator.py path-to-video-file`` <br>
This command will generate frames for a given video and save it in the ``frames`` directory. <br>
For example, <br>
``python frame_generator.py ./videos/video_1.mp4`` <br>


### (2) Instance Segmentation and Background Removal

``python seg_backrem.py`` <br>
The next step is the instance segmentation and background removal from each of the extracted frames in the ``frames`` directory. <br>
The background is changed to magenta color for having the contrast for the segmented image. <br>
This command will generate the segmented image for the frames and save them in ``fg-extract`` folder. <br>


### (3) Evaluation

``python test_modified.py path-to-video-file`` <br>
The final step is the evaluation of each of the segmented frames in the ``fg-extract`` directory. <br>
This will also generate the required plot of the predicted probability of heart attack at a specific time instant and also creates the corresponding json file. <br>
For example, <br>
``python test_modified.py ./videos/video_1.mp4`` <br>


#### README will be constantly updated with progress in the project. Stay tuned. :smile:

