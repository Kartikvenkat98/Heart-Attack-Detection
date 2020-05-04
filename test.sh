#!/bin/bash

#Replace the variables username and password with your Github username and password
GIT_REPO_URL="https://username:password@github.com/Kartikvenkat98/Heart-Attack-Detection.git"
REPO="Heart-Attack-Detection"

#Replace with the path to video. For eg., './videos/video_1.mp4'
VIDEO="./videos/video_6.mp4"
UIN_JSON="830000061.json"
UIN_PNG="830000061.png"

git clone $GIT_REPO_URL
cd $REPO
ls
#Replace this line with commands for running your test python file.
#!echo $VIDEO
python frame_generator.py $VIDEO
python seg_backrem.py
python test_modified.py $VIDEO
#If your test file is ipython file, uncomment the following lines and
#replace IPYTHON_NAME with your test ipython file.
#IPYTHON_NAME="test.ipynb"
#echo $IPYTHON_NAME
#jupyter notebook
#rename the generated timeLabel.json and figure with your UIN.
cp figures/timeLabel.json $UIN_JSON
cp figures/video_detect.png $UIN_PNG