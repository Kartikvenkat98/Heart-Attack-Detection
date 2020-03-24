from os import scandir
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import glob
import matplotlib.pyplot as plt
import re
import json


INPUT_PATH_TEST = "./dataset/test/"
MODEL_PATH = "./model/" + "model.h5"    


WIDTH, HEIGHT = 256, 256        
CLASS_COUNTING = False           
BATCH_SIZE = 32                 
CLASSES = ['00None', '01Infarct']   

print("Loading model from:", MODEL_PATH)
NET = load_model(MODEL_PATH)
NET.summary()

from subprocess import check_output
file_name = "./videos/video_1.mp4"
a = str(check_output('ffprobe -i  "'+file_name+'" 2>&1 |grep "Duration"',shell=True))
a = a.split(",")[0].split("Duration:")[1].strip()
h, m, s = a.split(':')
duration = int(h) * 3600 + int(m) * 60 + float(s)
print(duration)

def predict(file):
    
    x = load_img(file, target_size=(WIDTH, HEIGHT))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = NET.predict(x)
    result = array[0]
    answer = np.argmax(result)
    print(CLASSES[answer], result)
    return result[1]


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

pred = []
time = []
dictlist = []

for img in sorted(glob.glob('./fg-extract/*'), key=numericalSort):
    maxx = re.search(r'(\d+)', img).group(0)

print(maxx)

for img in sorted(glob.glob('./fg-extract/*'), key=numericalSort):
    try :
        k = predict(img)
        print(img)
        pred.append(k)
        l = re.search(r'(\d+)', img).group(0)
        t = int(l)/int(maxx) * duration
        time.append(t)
        dictlist.append([t, k])
            
    except Exception as e:
        print (e)

#print(time)

plt.plot(time, pred)
plt.xlabel('Time in seconds')
plt.ylabel('pred. prob. of heart attack')
#plt.show()
plt.savefig('./figures/video_detect.png')

#print(dictlist)
dicto = {"heart-attack" : str(dictlist)}
with open('./figures/timeLabel.json', 'w') as json_file:
  json.dump(dicto, json_file)