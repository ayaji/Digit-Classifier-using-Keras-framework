# -*- coding: utf-8 -*-
"""cnn.ipynb



Original file is located at
    https://colab.research.google.com/drive/1DQm8VratSn-a1rm6M8aQJqdAO05uJyCv
"""

import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from keras.utils import np_utils
# %matplotlib inline

np.random.seed(42)

from keras.datasets import mnist
mnist.load_data()

#Import data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
Y_test=y_test

#Sample showcase
import matplotlib.pyplot as plt


x_train=x_train[0:55000]
y_train=y_train[0:55000]

# reshape the data so as to fit the format of (samples, height, width, channels)
x_train = x_train.reshape(55000, 28, 28, 1).astype('float32')
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')

#One-hot encoding
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

x_train.shape

y_train.shape

import numpy as np
from keras.models import Sequential
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

"""#DELIVERABLE - 01 (MODEL BUILDING AND TRAINING THE CNN)

**Deliverable-01 Model:**
"""
print("DELIVERABLE - 01 (MODEL BUILDING AND TRAINING THE CNN)")
print("Deliverable-01 Model---------------------------------------------------------------------------------------------------------")
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

#Model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), kernel_regularizer=regularizers.l2(0.04), strides=(1,1), padding='valid', activation='relu', data_format='channels_last', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add(Conv2D(filters=64, kernel_size=(5,5), kernel_regularizer=regularizers.l2(0.04), strides=(2,2), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add(Dropout(rate=0.4,seed=3))

model.add(GlobalAveragePooling2D())
model.add(Dense(units=10,activation='softmax', kernel_regularizer=regularizers.l2(0.04)))
model.summary()

#Model Compilation
sgd = optimizers.sgd(lr=0.001)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[metrics.categorical_accuracy,precision,recall])

#Model fit
history=model.fit(x_train, y_train, epochs=5, batch_size=50,validation_data=(x_test, y_test))

# #Establishing connection to Google-drive
# from google.colab import drive 
# drive.mount('/g')
# from glob import glob 
# import os 
# os.chdir('/g/My Drive/apps')

#model.save('C:\\Users\\adityayaji\\Desktop\\Assignments\\CV\\AS4\\deliverable_1.h5')
#model.save_weights('C:\\Users\\adityayaji\\Desktop\\Assignments\\CV\\AS4\\my_model_weights_deliverable_1.h5')

model.save('..\\models\\deliverable_1.h5')
model.save('..\\models\\my_model_weights_deliverable_1.h5')
print("Model and their corresponding weights saved.....................................................................................")
"""# **REPORT THE TRAINING LOSS AND ACCURACY**"""
print("REPORTING THE TRAINED LOSS AND ACCURACY")
accuracy=history.history['categorical_accuracy']

loss=history.history['loss']

plt.figure(1,figsize=(7,5))
plt.plot(range(len(accuracy)),accuracy)
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Trained Accuracy Result')
plt.show()

plt.figure(2,figsize=(7,5))
plt.plot(range(len(loss)),loss)
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Trained Loss Result')
plt.show()

#Accuracy of final training step
print("Training Accuracy of the last epoch is",accuracy[4])

#Loss  of final training step
print("Training Loss of the last epoch is",loss[4])

"""# EVALUATION AND PREDICTION"""
print("EVALUATION AND PREDICTION")
val_recall=history.history['val_recall']
val_loss=history.history['val_loss']
val_precision=history.history['val_precision']
val_categorical_accuracy=history.history['val_categorical_accuracy']

plt.figure(3,figsize=(7,5))
plt.plot(range(len(val_categorical_accuracy)),val_categorical_accuracy)
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Result')
plt.show()

plt.figure(4,figsize=(7,5))
plt.plot(range(len(val_loss)),val_loss)
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.title('Test Loss Result')
plt.show()

plt.figure(5,figsize=(7,5))
plt.plot(range(len(val_precision)),val_precision)
plt.xlabel('Number of Epochs')
plt.ylabel('precision')
plt.title('Test Precision Result')
plt.show()

plt.figure(6,figsize=(7,5))
plt.plot(range(len(val_recall)),val_recall)
plt.xlabel('Number of Epochs')
plt.ylabel('recall')
plt.title('Test Recall Result')
plt.show()

# Test loss of the last epoch
print("Test loss of the last epoch is",val_loss[4])

# Test accuracy of the last epoch
print("Test accuracy of the last epoch is",val_categorical_accuracy[4])

# Test precision of the last epoch
print("Test precision of the last epoch is",val_precision[4])

# Test recall of the last epoch
print("Test recall of the last epochs",val_recall[4])

"""#Alternative way to evaluate and report the metrics of the last epoch:"""
print("Alternative way to evaluate and report the metrics of the last epoch")
score= model.evaluate(x_test, y_test, verbose=0)

score

model.metrics_names

print('Test loss:',score[0])

print('Test accuracy:',score[1])

print('Test precision:',score[2])

print('Test recall:',score[3])

"""# **Deliverable-02 (Parameter Tuning)**

**Deliverable_01_Variation_01 Model:**
"""
print("Deliverable-02 (Parameter Tuning)")
print("Deliverable_01_Variation_01 Model----------------------------------------------------------------------------------")
#Model
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(5,5), kernel_regularizer=regularizers.l2(0.04), strides=(1,1), padding='valid', activation='relu', data_format='channels_last', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

model.add(Conv2D(filters=64, kernel_size=(5,5), kernel_regularizer=regularizers.l2(0.04), strides=(2,2), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

model.add(Dropout(rate=0.5,seed=3))
model.add(GlobalAveragePooling2D())
model.add(Dense(units=10,activation='softmax', kernel_regularizer=regularizers.l2(0.04)))
model.summary()

#Model Compilation
adam = optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy,precision,recall])

#Model fit
history=model.fit(x_train, y_train, epochs=3, batch_size=50,validation_data=(x_test, y_test))

model.save('..\\models\\deliverable_2_variation_1.h5')
model.save_weights('..\\models\\my_model_weights_variation_1.h5')
print("Model and their corresponding weights saved.....................................................................................")
"""**Metrics Evaluation of Variation_01:**"""
print("Metrics Evaluation of Variation_01---------------------------------------------------------------------------------------------")
score_train_variation_01= model.evaluate(x_train, y_train, batch_size=100)

print(score_train_variation_01)

print(model.metrics_names)

score_test_variation_01= model.evaluate(x_test, y_test, batch_size=100)

print(score_test_variation_01)

print(model.metrics_names)

"""**Deliverable_01_Variation_02 Model:**"""
print("Deliverable_01_Variation_02 Model---------------------------------------------------------------------------------------------")
#Model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), kernel_regularizer=regularizers.l2(0.04), strides=(1,1), padding='valid', activation='relu', data_format='channels_last', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

model.add(Conv2D(filters=64, kernel_size=(5,5), kernel_regularizer=regularizers.l2(0.04), strides=(2,2), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

model.add(Conv2D(filters=100, kernel_size=(7,7), kernel_regularizer=regularizers.l2(0.04), strides=(1,1), padding='valid', activation='relu', data_format='channels_last', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

model.add(Dropout(rate=0.5,seed=3))


model.add(GlobalAveragePooling2D())
model.add(Dense(units=10,activation='softmax', kernel_regularizer=regularizers.l2(0.04)))
model.summary()
#Model Compilation
adam = optimizers.Adam(lr=0.001) 
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_accuracy,precision,recall])
#Model fit
history=model.fit(x_train, y_train, epochs=5, batch_size=75,validation_data=(x_test, y_test))

model.save('..\\models\\deliverable_2_variation_2.h5')
model.save_weights('..\\models\\my_model_weights_variation_2.h5')
print("Model and their corresponding weights saved.....................................................................................")
"""**Metrics Evaluation of Variation_02:**"""
print("Metrics Evaluation of Variation_02:")
score_train_variation_02= model.evaluate(x_train, y_train, batch_size=75)

print(score_train_variation_02)

print(model.metrics_names)

score_test_variation_02= model.evaluate(x_test, y_test, batch_size=100)

print(score_test_variation_02)

print(model.metrics_names)

"""**Deliverable_01_Variation_03 Model:**"""
print("Deliverable_01_Variation_03 Model")
print("Deliverable_01_Variation_03 Model--------------------------------------------------------------------------------------------------------------------------------")
#Model
model = Sequential()

model.add(Conv2D(filters=100, kernel_size=(5,5), kernel_regularizer=regularizers.l2(0.04), strides=(1,1), padding='valid', activation='relu', data_format='channels_last', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

model.add(Conv2D(filters=100, kernel_size=(5,5), kernel_regularizer=regularizers.l2(0.04), strides=(2,2), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1), strides=(2,3)))

#model.add(Conv2D(filters=100, kernel_size=(7,7), kernel_regularizer=regularizers.l2(0.04), strides=(1,1), padding='valid', activation='relu', data_format='channels_last', input_shape=(28,28,1)))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

model.add(Dropout(rate=0.6,seed=3))


model.add(GlobalAveragePooling2D())
model.add(Dense(units=10,activation='softmax', kernel_regularizer=regularizers.l2(0.04)))
model.summary()
#Model Compilation
adam = optimizers.Adam(lr=0.001) 
model.compile(loss='kullback_leibler_divergence', optimizer=adam, metrics=[metrics.categorical_accuracy,precision,recall])
#Model fit
history=model.fit(x_train, y_train, epochs=5, batch_size=200,validation_data=(x_test, y_test))

model.save('..\\models\\deliverable_2_variation_3.h5')
model.save_weights('..\\models\\my_model_weights_variation_3.h5')
print("Model and their corresponding weights saved.....................................................................................")
"""Metrics Evaluation of Variation_3:"""
print("Metrics Evaluation of Variation_3:")
score_train_variation_3= model.evaluate(x_train, y_train, batch_size=75)

print(score_train_variation_3)

print(model.metrics_names)

score_test_variation_3= model.evaluate(x_test, y_test, batch_size=100)

print(score_test_variation_3)

print(model.metrics_names)

"""**Deliverable_01_Variation_04 Model:**"""
print("Deliverable_01_Variation_04 Model-------------------------------------------------------------------------------------------------------------------------------")
#Model
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(5,5), kernel_regularizer=regularizers.l2(0.04), strides=(1,1), padding='valid', activation='relu', data_format='channels_last', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))

model.add(Conv2D(filters=64, kernel_size=(5,5), kernel_regularizer=regularizers.l2(0.04), strides=(2,2), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model.add(Dropout(rate=0.25,seed=3))


model.add(GlobalAveragePooling2D())
model.add(Dense(units=10,activation='softmax', kernel_regularizer=regularizers.l2(0.04)))
model.summary()
#Model Compilation
rms = optimizers.RMSprop(lr=0.0001) 
model.compile(loss='mean_squared_error', optimizer=rms, metrics=[metrics.categorical_accuracy,precision,recall])
#Model fit
history=model.fit(x_train, y_train, epochs=7, batch_size=300,validation_data=(x_test, y_test))

model.save('..\\models\\deliverable_2_variation_4.h5')
model.save_weights('..\\models\\my_model_weights_variation_4.h5')
print("Model and their corresponding weights saved.....................................................................................")
"""Metrics Evaluation of Variation_4:"""
print("Metrics Evaluation of Variation_4:")
score_train_variation_4= model.evaluate(x_train, y_train, batch_size=75)

print(score_train_variation_4)

print(model.metrics_names)

score_test_variation_3= model.evaluate(x_test, y_test, batch_size=100)

print(score_test_variation_3)

print(model.metrics_names)