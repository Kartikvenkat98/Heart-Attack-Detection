# Heart-Attack-Detection

## Author
**Kartik Venkataraman**


## Preprocessing

(1) mkdir dataset && mkdir videos && mkdir dataset_pose && mkdir videos_pose && mkdir json_pose<br>
(2) Download dataset from [here](https://drive.google.com/drive/folders/16HhfMovQMS8iMgBVFRwJ4tmpOwmVzprG?usp=sharing) by logging with TAMU email and put them under ``dataset`` folder <br>
(3) Download videos from [here](https://drive.google.com/drive/folders/1ka5vGFS09oeEejoiIFoFbgdrHPmMXCxr?usp=sharing) by logging with TAMU email and put them under ``videos`` folder <br>


## Installing Openpose

[Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) is a tool for generating facial/body landmarks for humans in videos and images. <br>
Install OpenPose in your Google Colab working directory by following the instructions found [here](https://colab.research.google.com/github/tugstugi/dl-colab-notebooks/blob/master/notebooks/OpenPose.ipynb). <br>


## Running Openpose on Images

``./build/examples/openpose/openpose.bin --image_dir path-to-images-directory --write-images path-to-rendered-images-directory --write_json path-to-json-pose-directory --display 0`` <br>
This command will generate body landmarks for all the images in the given directory and save the rendered images in ``dataset_pose`` and json in ``json_pose`` directory. <br>
For example, <br>
``./build/examples/openpose/openpose.bin --image_dir ../dataset/train --write_images ../dataset_pose/train --write_json ../json_pose/train --display 0`` <br>


## Training the model

``python train.py`` <br>
This command will train the CNN on the images in the ``train`` folder inside the ``dataset`` directory and save the trained model as model.h5 under ``model`` folder. <br>
For training on the Openpose rendered images, change the path to ``train`` folder inside the ``dataset_pose`` directory and save the model as model_pose.h5 under ``model`` folder. <br>

Alternatively, you can directly use the trained uploaded model.


## Testing the model

``python test.py`` <br>
This command will predict infraction on the static images in the ``test`` folder inside the ``dataset`` directory. <br>
For testing on the Openpose rendered images, change the path to ``test`` folder inside the ``dataset_pose``. <br>
The initial model shows 91.75% accuracy. <br>
The model trained on Openpose rendered images shows 93.25% accuracy. <br>


## Detection in Videos

There are four steps for detecting possible heart-attacks from videos.

### (1) Generating body landmarks on the video using Openpose

``./build/examples/openpose/openpose.bin --hand --video path-to-video-file --write_video path-to-video-pose --write_json path-to-json-output --display 0``<br>
This command will generate body landmarks for the whole video and save the rendered video in the video_pose directory. <br>
For example, <br>
``./build/examples/openpose/openpose.bin --hand --video ../videos/video_1.mp4 --write_video ../videos_pose/video_1.mp4 --write_json ../json_pose/video_1 --display 0`` <br>

### (2) Frame Generation from videos

``python frame_generator.py path-to-video-file`` <br>
This command will generate frames for a given video and save it in the ``frames`` directory. <br>
For example, <br>
``python frame_generator.py ./videos_pose/video_1.mp4`` <br>


### (3) Instance Segmentation and Background Removal

``python seg_backrem.py`` <br>
The next step is the instance segmentation and background removal from each of the extracted frames in the ``frames`` directory. <br>
The background is changed to magenta color for having the contrast for the segmented image. <br>
This command will generate the segmented image for the frames and save them in ``fg-extract`` folder. <br>


### (4) Evaluation

``python test_modified.py path-to-video-file`` <br>
The final step is the evaluation of each of the segmented frames in the ``fg-extract`` directory. <br>
This will also generate the required plot of the predicted probability of heart attack at a specific time instant and also creates the corresponding json file. <br>
For example, <br>
``python test_modified.py ./videos_pose/video_1.mp4`` <br>


#### README will be constantly updated with progress in the project. Stay tuned. :smile:

