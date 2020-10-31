import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn as sk

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from random import shuffle
from PIL import Image

plt.style.use("ggplot")

'''
Each image will be read in from the directories containing N = 20,000 images and represented as the feature vector Xi. 
Each image is 227 X 227 and each pixel is greyscale (0-255) thus xi is 227 X 227 X 256 = 13191424 elements long (consider revising)
and will ave the associated class label yi there are a total of K=2 class labels (positive and negative)   

'''

def main():

    data = []
    Positives = os.listdir("Concrete_Crack_Images_for_Classification/Positive")
    Negatives = os.listdir("Concrete_Crack_Images_for_Classification/Negative")

    for img in Positives: # This will take a while
        try:
            # Read the images
            print("Adding Positive Image ", img)
            bgr_image = cv2.imread("Concrete_Crack_Images_for_Classification/Positive/" + str(img)) 
            gry_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY) # Convert image to greyscale
            data.append([gry_img, 1]) # Label 1 for positive
        except AttributeError:
            print("")

    for img in Negatives: # This will take a while
        try:
            print("Adding Negative Image ", img)
            # Read the images
            bgr_image = cv2.imread("Concrete_Crack_Images_for_Classification/Negative/" + str(img)) 
            gry_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY) # Convert image to greyscale
            data.append([gry_img, 0]) # Label 0 for negative
        except AttributeError:
            print("")

    shuffle(data) # Shuffle the data 
    feature_vector_X = [data[vec][0] for vec in range(len(data))]
    data_labels_y = [data[label][1] for label in range(len(data))]

    X_train, X_test, y_train, y_test = train_test_split(feature_vector_X, data_labels_y, random_state=0)

if __name__ == "__main__":
    main()