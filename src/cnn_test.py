#Importing all the required libraries
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import sys
import cv2
import numpy as np
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras import regularizers
from keras import metrics
from keras.utils.np_utils import to_categorical
from keras import optimizers
from scipy import misc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.utils import class_weight
from keras import backend as K
from keras.datasets import mnist
from keras.models import load_model

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall



home='..\\models\\deliverable_1.h5'
model=load_model(home,custom_objects={'precision':precision,'recall':recall})
model.load_weights('..\\models\\my_model_weights_deliverable_1.h5')
print("Model loaded successfully......................................")
#Inverting the gray scale
def inverte(imagem):
    imagem = (255-imagem)
    return imagem

def process_image():
    path=input("enter path of the Image :")
    if(path=="q"):
        print("Terminating the function")
        sys.exit(0)
    else:
    #Reading image from specified path
        img=cv2.imread("%s" %(path),0)

        img=inverte(img)

        #Gaussian blur
        blur = cv2.GaussianBlur(img,(9,9),0)

        #Adaptive threshold
        ret,thresh1 = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)

        #Resizing image to fit the image requirement of the model
        im=cv2.resize(thresh1,(28,28),interpolation= cv2.INTER_AREA).astype('float32')
        imgs=np.expand_dims(np.expand_dims(im,axis=2),axis=0)

        cv2.imshow('Original Image',img)
        cv2.imshow('Binary Image',thresh1)

        #Predicting using the model
        im_predict=model.predict(imgs)

        #Identifyig the digit in the image
        im_predict=np.argmax(im_predict,axis=1)

        print("Prediction")
        print(im_predict)

        #Classifying it as even or odd
        if(im_predict%2==0):
            print("Image has an even number")
        else:
            print("Image has an odd number")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

import time,msvcrt
def function():
    while True:
        time.sleep(3)
        
        if (msvcrt.kbhit()) and (msvcrt.getch()==b'q'):
            print("'q' is pressed to terminate the function")
            break
        else:
            process_image()
    
function()
