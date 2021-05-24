"""
1) Real time face recognition
2) doesn't allow more then one face to add at time
3) If face is already added then doesn't allow to add
4) 
"""

import cv2
import numpy as np
from PIL import Image
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
try:
    recognizer.read('trainer/trainer.yml')
    savedModel = True
except:
    savedModel = False
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
path = 'dataset'

font = cv2.FONT_HERSHEY_SIMPLEX
names = []

file = open("names.txt","r")
for i in file.readlines():
    ii = i.strip("\n")
    names.append(ii)
print(names)

count = 0 

cam = cv2.VideoCapture(0)
cam.set(3, 340)
cam.set(4, 280)

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

onAction = False

def getDatasetId():
    try:
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        for imagePath in imagePaths:
            id = int(os.path.split(imagePath)[-1].split(".")[1])
        return id
    except:
        return 0

def writer(faces , ids):
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml')
    print("Training Complete")

def Return_Faces_and_ids(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') 
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = faceCascade.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
idd = ""
while True:
    ret, frames =cam.read()
##    frames = cv2.flip(frames, -1)

    gray = cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 5,minSize = (int(minW), int(minH)),)

    for(x,y,w,h) in faces:
        cv2.rectangle(frames, (x,y), (x+w,y+h), (0,255,250), 2)
        idd = "unknown"
        if savedModel == True:
            if onAction == False:
                idd, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                if (confidence < 100):
                    idd = names[idd]
                    confidence = round(100 - confidence)
    ##                print(str(idd),confidence)
##                    print(str(idd))
                else:
                    idd = "unknown"
                    confidence = round(100 - confidence)
##                    print(str(idd))
                cv2.putText(frames, str(confidence), (x+5,y+h-5), font, 1, (255,100,0), 1)
                cv2.putText(frames, str(idd), (x+5,y-5), font, 1, (255,20,200), 2)

            if onAction == True:
                count +=1
                print(count," Picture")
                cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                if count >= 30:
                    A,B = Return_Faces_and_ids(path)
                    writer(A,B)
                    names.append(face_name)
                    print(names)
                    updateNames = open("names.txt","a+")
##                    updateNames.write("\n")
                    updateNames.write(str(face_name))
                    updateNames.write("\n")
                    updateNames.close()
                    onAction = False
                    count = 0
                    recognizer.read('trainer/trainer.yml')
                    print("Ready To use")

            
              

    cv2.imshow("Output",frames)
##    print(str(idd))
    k = cv2.waitKey(10) & 0xff 
    if k == ord("q"):
        if len(faces) >= 1 and len(faces) < 2:
            if savedModel == True:
                if (idd in names):
                    print("face Already exist in database")
                else:
                    print("Training Started")
                    face_id = getDatasetId() + 1
                    face_name = input("Enter your name: ")
                    print(face_id," = ",face_name)
                    savedModel = True
                    onAction = True
            else:   
                print("Training Started")
                face_id = getDatasetId() + 1
                face_name = input("Enter your name: ")
                print(face_id," = ",face_name)
                savedModel = True
                onAction = True
        elif len(faces) >= 2:
            print("Too Many face detection, we required one at time.")
        else:
            print("No Face to Add")
    if k == 27:
        break

print("End")
cam.release()
cv2.destroyAllWindows()
