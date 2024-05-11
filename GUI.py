from tkinter import *
import ctypes,os
from PIL import ImageTk, Image
import tkinter.messagebox as tkMessageBox
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import ctypes
from keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
from playsound import playsound

from TheLazyCoder import social_distancing_config as config
from TheLazyCoder.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
from keras.models import load_model
import keras.utils as image

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")



mymodel=load_model('C:/Users/hp/OneDrive/Desktop/PROJECT/face mask/mymodel.h5')
face_cascade=cv2.CascadeClassifier('C:/Users/hp/OneDrive/Desktop/PROJECT/face mask/haarcascade_frontalface_default.xml')



#################################################################

home = Tk()
home.title("FaceMask & Social Distancing Detection")

img = Image.open("images/home.png")
img = ImageTk.PhotoImage(img)
panel = Label(home, image=img)
panel.pack(side="top", fill="both", expand="yes")
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
lt = [w, h]
a = str(lt[0]//2-450)
b= str(lt[1]//2-320)
home.geometry("900x653+"+a+"+"+b)
home.resizable(0,0)
file = ''


def Exit():
    global home
    result = tkMessageBox.askquestion(
        "FaceMask & Social Distancing", 'Are you sure you want to exit?', icon="warning")
    if result == 'yes':
        home.destroy()
        exit()
    else:
        tkMessageBox.showinfo(
            'Return', 'You will now return to the main screen')

 
def browse():
    
    global file,l1
    try:
        l1.destroy()
    except:
        pass
    file = askopenfilename(filetypes=(("Video files", "*.mp4;*.flv;*.avi;*.mkv"),("All files", "*.*") ))
    #askopenfilename(initialdir=os.getcwd(), title="Select Image", filetypes=( ("images", ".png"),("images", ".jpg"),("images", ".mp4")))
    
    vs = cv2.VideoCapture(file) 
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
    
        face=face_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=4)
        for(x,y,w,h) in face:
                face_frame = frame[y:y+h, x:x+w]
                cv2.imwrite('C:/Users/hp/OneDrive/Desktop/PROJECT/face mask/temp.jpg',face_frame)
                test_image=image.load_img('C:/Users/hp/OneDrive/Desktop/PROJECT/face mask/temp.jpg',target_size=(150,150,3))
                test_image=image.img_to_array(test_image)
                test_image=np.expand_dims(test_image,axis=0)
                pred=mymodel.predict(test_image)[0][0]
                if pred==1:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
                    cv2.putText(frame,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                else:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
                    cv2.putText(frame,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
  
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
                break

        # resize the frame and then detect people (and only people) in it
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln,
                personIdx=LABELS.index("person"))

        # initialize the set of indexes that violate the minimum social
        # distance
        violate = set()

        # ensure there are *at least* two people detections (required in
        # order to compute our pairwise distance maps)
        if len(results) >= 2:
                # extract all centroids from the results and compute the
                # Euclidean distances between all pairs of the centroids
                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")

                # loop over the upper triangular of the distance matrix
                for i in range(0, D.shape[0]):
                        for j in range(i + 1, D.shape[1]):
                                # check to see if the distance between any two
                                # centroid pairs is less than the configured number
                                # of pixels
                                if D[i, j] < config.MIN_DISTANCE:
                                        # update our violation set with the indexes of
                                        # the centroid pairs
                                        violate.add(i)
                                        violate.add(j)
                                        messagebox.showwarning("showwarning","You are violating the social distancing rules please maintain 1feet distance")

        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
                # extract the bounding box and centroid coordinates, then
                # initialize the color of the annotation
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)

                # if the index pair exists within the violation set, then
                # update the color
                if i in violate:
                        color = (0, 0, 255)

                # draw (1) a bounding box around the person and (2) the
                # centroid coordinates of the person,
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)
                

        # draw the total number of social distancing violations on the
        # output frame
        text = "Social Distancing Violations: {}".format(len(violate))
        
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF 
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        
def live():
    vs = cv2.VideoCapture(0) 
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
    
        face=face_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=4)
        for(x,y,w,h) in face:
                face_frame = frame[y:y+h, x:x+w]
                cv2.imwrite('C:/Users/hp/OneDrive/Desktop/PROJECT/face mask/temp.jpg',face_frame)
                test_image=image.load_img('C:/Users/hp/OneDrive/Desktop/PROJECT/face mask/temp.jpg',target_size=(150,150,3))
                test_image=image.img_to_array(test_image)
                test_image=np.expand_dims(test_image,axis=0)
                pred=mymodel.predict(test_image)[0][0]
                if pred==1:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
                    cv2.putText(frame,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                else:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
                    cv2.putText(frame,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
  
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
                break

        # resize the frame and then detect people (and only people) in it
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln,
                personIdx=LABELS.index("person"))

        # initialize the set of indexes that violate the minimum social
        # distance
        violate = set()

        # ensure there are *at least* two people detections (required in
        # order to compute our pairwise distance maps)
        if len(results) >= 2:
                # extract all centroids from the results and compute the
                # Euclidean distances between all pairs of the centroids
                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")

                # loop over the upper triangular of the distance matrix
                for i in range(0, D.shape[0]):
                        for j in range(i + 1, D.shape[1]):
                                # check to see if the distance between any two
                                # centroid pairs is less than the configured number
                                # of pixels
                                if D[i, j] < config.MIN_DISTANCE:
                                        # update our violation set with the indexes of
                                        # the centroid pairs
                                        violate.add(i)
                                        violate.add(j)
                                        messagebox.showwarning("showwarning","You are violating the social distancing rules please maintain 1feet distance")

        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
                # extract the bounding box and centroid coordinates, then
                # initialize the color of the annotation
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)

                # if the index pair exists within the violation set, then
                # update the color
                if i in violate:
                        color = (0, 0, 255)

                # draw (1) a bounding box around the person and (2) the
                # centroid coordinates of the person,
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)
                


        # draw the total number of social distancing violations on the
        # output frame
        text = "Social Distancing Violations: {}".format(len(violate))


        cv2.putText(frame, text, (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF 
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break



def about():
    about = Toplevel()
    about.title("FaceMask & Social Distancing")
    img = Image.open("images/about.png")
    img = ImageTk.PhotoImage(img)
    panel = Label(about, image=img)
    panel.pack(side="top", fill="both", expand="yes")
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
    lt = [w, h]
    a = str(lt[0]//2-450)
    b= str(lt[1]//2-320)
    about.geometry("900x653+"+a+"+"+b)
    about.resizable(0,0)
    about.mainloop()
    
photo = Image.open("images/1.png")
img2 = ImageTk.PhotoImage(photo)
b1=Button(home, highlightthickness = 0, bd = 0,activebackground="#2b4b47", image = img2,command=browse)
b1.place(x=0,y=209)

photo = Image.open("images/2.png")
img3 = ImageTk.PhotoImage(photo)
b2=Button(home, highlightthickness = 0, bd = 0,activebackground="#2b4b47", image = img3, command=live)
b2.place(x=0,y=282)

photo = Image.open("images/3.png")
img4 = ImageTk.PhotoImage(photo)
b3=Button(home, highlightthickness = 0, bd = 0,activebackground="#2b4b47", image = img4,command=about)
b3.place(x=0,y=354)

photo = Image.open("images/4.png")
img5 = ImageTk.PhotoImage(photo)
b4=Button(home, highlightthickness = 0, bd = 0,activebackground="#2b4b47", image = img5,command=Exit)
b4.place(x=0,y=426)

home.mainloop()
