#This app ties together all our work

#------------ Import libraries

#COMMON
import pandas as pd #DF work
import numpy as np #Functions
import matplotlib.pyplot as plt #Visualizations
import requests
import altair as alt #Visualizations
import io #Buffers for DF's
from io import BytesIO #BytesIO
from io import StringIO #StringIO
import http.client #API
import os #operating system functions
from PIL import Image #open pictures
from pathlib import Path #path function
from scipy.io import loadmat #load .mat files
import datetime #dates and time stuff
import json

#uploading/file management
from tempfile import NamedTemporaryFile

#ML/Computer Vision
import pickle
import cv2 #computer vision
import tensorflow as tf #tensorflow
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator #generate
from keras.models import Sequential #sequential model
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D #model functions
from keras import optimizers #model optimizers such as Adam and learning rates

#Deployment
import streamlit as st #app deployment

#Introduction

st.title('Car License Plate Detector :car:')


#header for the project
st.header('What does it do?')
st.markdown('This is an app that _regonizes_ and **detects** license plates in images, then return an image of the car with license plate.')

#------------ Data input
st.subheader('Input a picture')
st.markdown('Upload your picture in the box below, or take a picture with your phone')

#upload a picture
uploaded_file = st.file_uploader("Upload your picture (only .jpg)", type=["jpg"])
if uploaded_file is not None:
    st.write("Original Image")
    st.image(uploaded_file, caption="Uploaded Image")

    # Convert image to cv2 format
    bytes_data = uploaded_file.getvalue()
    opencv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Loads the data required for detecting the license plates from cascade classifier.
    plate_cascade = cv2.CascadeClassifier('indian_license_plate.xml')
    #identifying the plate
    def detect_plate(img, text=''):
        plate_img = img.copy()
        roi = img.copy()
        plates = []  # Store all detected plates

        plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.2, minNeighbors=7)
        for (x, y, w, h) in plate_rect:
            roi_ = roi[y:y+h, x:x+w, :]
            plate = roi[y:y+h, x:x+w, :]
            cv2.rectangle(plate_img, (x+2, y), (x+w-3, y+h-5), (51, 181, 155), 3)
            plates.append(plate)  # Store each detected plate

        if text != '':
            plate_img = cv2.putText(plate_img, text, (x-w//2, y-h//2),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.5, (51, 181, 155), 1, cv2.LINE_AA)

        return plate_img, plates
    
    #image reading
    def display(img_, title=''):
        img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize=(10,6))
        ax = plt.subplot(111)
        ax.imshow(img)
        plt.axis('off')
        plt.title(title)
        plt.show()

    img =opencv_image

    # Getting plate prom the processed image
    output_img ,plates= detect_plate(img)
    for i, plate in enumerate(plates):
        display(plate, f'License Plate {i+1}')
    
    # Match contours to license plate or character template
    def find_contours(dimensions, img) :

        # Find all contours in the image
        cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Retrieve potential dimensions
        lower_width = dimensions[0]
        upper_width = dimensions[1]
        lower_height = dimensions[2]
        upper_height = dimensions[3]

        # Check largest 5 or  15 contours for license plate or character respectively
        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

        ii = cv2.imread('contour.jpg')

        x_cntr_list = []
        target_contours = []
        img_res = []
        for cntr in cntrs :
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
            intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

            # checking the dimensions of the contour to filter out the characters by contour's size
            if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
                x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

                char_copy = np.zeros((44,24))
                # extracting each character using the enclosing rectangle's coordinates.
                char = img[intY:intY+intHeight, intX:intX+intWidth]
                char = cv2.resize(char, (20, 40))

                cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
                plt.imshow(ii, cmap='gray')

                # Make result formatted for classification: invert colors
                char = cv2.subtract(255, char)

                # Resize the image to 24x44 with black border
                char_copy[2:42, 2:22] = char
                char_copy[0:2, :] = 0
                char_copy[:, 0:2] = 0
                char_copy[42:44, :] = 0
                char_copy[:, 22:24] = 0

                img_res.append(char_copy) # List that stores the character's binary image (unsorted)

        # Return characters on ascending order with respect to the x-coordinate (most-left character first)

        plt.show()
        # arbitrary function that stores sorted list of character indeces
        indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
        img_res_copy = []
        for idx in indices:
            img_res_copy.append(img_res[idx])# stores character images according to their index
        img_res = np.array(img_res_copy)
        return img_res
    # Find characters in the resulting images
    def segment_characters(image) :

        # Preprocess cropped license plate image
        img_lp = cv2.resize(image, (333, 75))
        img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
        _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_binary_lp = cv2.erode(img_binary_lp, (3,3))
        img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

        LP_WIDTH = img_binary_lp.shape[0]
        LP_HEIGHT = img_binary_lp.shape[1]

        # Make borders white
        img_binary_lp[0:3,:] = 255
        img_binary_lp[:,0:3] = 255
        img_binary_lp[72:75,:] = 255
        img_binary_lp[:,330:333] = 255

        # Estimations of character contours sizes of cropped license plates
        dimensions = [LP_WIDTH/6,
                        LP_WIDTH/2,
                        LP_HEIGHT/10,
                        2*LP_HEIGHT/3]
        plt.imshow(img_binary_lp, cmap='gray')
        plt.show()
        cv2.imwrite('contour.jpg',img_binary_lp)

        # Get contours within cropped license plate
        char_list = find_contours(dimensions, img_binary_lp)

        return char_list
    #characters
    char = segment_characters(plate)
    for i in range(len(char)):
        plt.subplot(1, len(char), i+1)
        plt.imshow(char[i], cmap='gray')
        plt.axis('off')
    #load model
    loaded_model_2 = tf.keras.models.load_model('license_20.h5')



    # Define the fix_dimension function
    def fix_dimension(img):
        # Check the number of channels in the input image
        if len(img.shape) == 2:  # Grayscale image
            new_img = np.zeros((28, 28, 3))
            for i in range(3):
                new_img[:, :, i] = img
        elif len(img.shape) == 3 and img.shape[2] == 3:  # Color image (RGB)
            new_img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        else:
            raise ValueError("Unsupported image format. Only grayscale and RGB images are supported.")
        return new_img

    # Define the show_results function
    def show_results():
        dic = {}
        characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i, c in enumerate(characters):
            dic[i] = c

        output = []
        for i, ch in enumerate(char):  # iterating over the characters
            img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
            img = fix_dimension(img_)
            img = img.reshape(1, 28, 28, 3)  # preparing image for the model
            probabilities = loaded_model_2.predict(img)  # predict probabilities for each class

            # Get the predicted class index (index of the highest probability)
            predicted_class_index = np.argmax(probabilities, axis=1)[0]

            # Get the corresponding character from the dictionary
            character = dic[predicted_class_index]

            output.append(character)  # storing the result in a list

        plate_number2= ''.join(output)

        return plate_number2
    
    plate_number2 = show_results()
    output_img, plate = detect_plate(img, plate_number2)
    st.image(output_img)
    st.subheader('detected plate number')
    st.write(plate_number2)
