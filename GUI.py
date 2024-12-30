import cv2
import cvzone
from tkinter import *
from tkinter import Label, Tk, Button
from PIL import Image, ImageTk
from os import listdir
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
from keras.models import load_model
import sys

file = "weights-44-0.93.hdf5"
# Dinh nghia class
# class_name = ['0k','10k','20k','50k','100k','200k','500k']
class_name = ['00000', '5000']


def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    # Dong bang cac layer
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # Tao model
    input = Input(shape=(128, 128, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    # Them cac layer FC va Dropout
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    # x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    # x = Dropout(0.5)(x)
    x = Dense(7, activation='softmax', name='predictions')(x)

    # Compile
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy',
                     optimizer='adam', metrics=['accuracy'])
    return my_model


def Webcam():
    # Load weights model da train
    my_model = get_model()
    my_model.load_weights(file)

    cap = cv2.VideoCapture(0)
    while (True):
        ret, image_org = cap.read()
        if not ret:
            continue
        image_org = cv2.resize(image_org, dsize=None, fx=0.5, fy=0.5)
        # Resize
        image = image_org.copy()
        image = cv2.resize(image, dsize=(128, 128))
        image = image.astype('float')*1./255
        # Convert to tensor
        image = np.expand_dims(image, axis=0)
        # Predict
        predict = my_model.predict(image)
        print("This picture is: ",
              class_name[np.argmax(predict[0])], (predict[0]))
        print(np.max(predict[0], axis=0))
        if (np.max(predict) >= 0.8) and (np.argmax(predict[0]) != 0):

            # Show image
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1.5
            color = (0, 255, 0)
            thickness = 2

            cv2.putText(image_org, class_name[np.argmax(
                predict)], org, font, fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow("Picture", image_org)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


root = Tk()
root.title('NHOM - 06')
root.geometry("1280x720")
root.iconbitmap('logo1.ico')
load = Image.open('BRs.png')
render = ImageTk.PhotoImage(load)
img = Label(root, image=render)
img.place(x=0, y=0)
b = Button(root, text="OpenCAM", command=Webcam, pady=10)
b.place(x=1100, y=600)
root.mainloop()
