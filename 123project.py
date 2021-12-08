import cv2
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

X = np.load("image.npz")["arr_0"]
Y = pd.read_csv("122projectdata1.csv")["labels"]
print(pd.Series(Y).value_counts())
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses = len(classes)

if(not os.environ.get("PYTHONHTTPSVERIFY","") and getattr(ssl,"_create_unverified_context",None)):
    ssl._create_default_https_context = ssl._create_unverified_context

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,random_state = 9,test_size = 2500,train_size = 7500)
Xtrainscale = Xtrain/255.0
Xtestscale = Xtest/255.0
clf = LogisticRegression(solver = "saga",multi_class = "multinomial").fit(Xtrainscale,Ytrain)
Ypred = clf.predict(Xtestscale)
accuracy = accuracy_score(Ytest,Ypred)
print(accuracy)


cap = cv2.VideoCapture(0)##0 is for device index
while (True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape()
        upperleft = (int(width/2-56),int(height/2-56))
        bottomright = (int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upperleft,bottomright,(0,255,0),2)
        roi = gray[upperleft[1]:bottomright[1],upperleft[0]:bottomright[0]]##roi is region of interest
        im_pil = Image.fromarray(roi) 
        image_bw = im_pil.convert("L")
        image_bw_resize = image_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_resize_inverted = PIL.ImageOps.invert(image_bw_resize)
        pixelfilter = 20
        min_pixel = np.percentile(image_bw_resize_inverted,pixelfilter)
        image_bw_resize_inverted_scale = np.clip(image_bw_resize_inverted-min_pixel,0,255)
        max_pixel = np.max(image_bw_resize_inverted)
        image_bw_resize_inverted_scale = np.asarray(image_bw_resize_inverted_scale)/max_pixel##converting into array
        test_sample = np.array(image_bw_resize_inverted_scale).reshape(1,784)
        test_pred = clf.predict(test_sample)
        print("predicted class is: ",test_pred)
        cv2.imshow("frame",gray)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()                
    


