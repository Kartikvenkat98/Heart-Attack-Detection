import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
import glob
import matplotlib.pyplot as plt
import re
import json
import sys
from subprocess import check_output
from keras.models import model_from_json

model = model_from_json(open("./model/model_fer.json", "r").read())
model.load_weights('./model/model_fer.h5')

MODEL_PATH = "./model/" + "model.h5"    


WIDTH, HEIGHT = 256, 256        
CLASS_COUNTING = False           
BATCH_SIZE = 32                 
CLASSES = ['00None', '01Infarct']   

print("Loading model from:", MODEL_PATH)
NET = load_model(MODEL_PATH)
NET.summary()

def emotion_analysis(emotions):

    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()
    return emotions

def face_analysis(file):

  img = load_img(file, grayscale=True, target_size=(48, 48))

  x = img_to_array(img)
  x = np.expand_dims(x, axis = 0)

  x /= 255

  custom = model.predict(x)
  #print(custom[0][5])
  return emotion_analysis(custom[0])


file_name = sys.argv[1]
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

count = 0
flag = False

for img in sorted(glob.glob('./fg-extract/*'), key=numericalSort):
    try :
        k = predict(img)
        print(img)
        if k >= 0.5:
          '''if not flag:
            emo = face_analysis(img)
            print(emo)
            if emo[0] >= 0.25 or emo[1] >= 0.25 or emo[2] >= 0.25 or emo[4] >= 0.25 or emo[6] >= 0.25:
              if emo[3] <= 0.25 and emo[5] <= 0.25:
                count += 1
              else:
                k = 0.0001
            '''
          count += 1
            #else:
              #k = 0.0001
          if count >= 30:
            flag = True
        if flag:
          emo = face_analysis(img)
          print(emo)
          if emo[0] >= 0.25 or emo[1] >= 0.25 or emo[2] >= 0.25 or emo[4] >= 0.25 or emo[6] >= 0.25:
            k = 0.9999
        pred.append(k)
        l = re.search(r'(\d+)', img).group(0)
        t = int(l)/int(maxx) * duration
        time.append(t)
        dictlist.append([t, k])
            
    except Exception as e:
        print (e)

#print(time)

dir_path = "./figures/"

if not os.path.exists(dir_path):
  os.mkdir(dir_path)

plt.plot(time, pred)
plt.xlabel('Time in seconds')
plt.ylabel('pred. prob. of heart attack')
#plt.show()
plt.savefig(dir_path + 'video_detect.png')

#print(dictlist)
dicto = {"heart-attack" : str(dictlist)}
with open(dir_path + 'timeLabel.json', 'w') as json_file:
  json.dump(dicto, json_file)