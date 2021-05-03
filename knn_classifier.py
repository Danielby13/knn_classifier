# Authors:
# Inbal Altan, 201643459
# Daniel Ben Yair, 204469118

import cv2
import numpy as np
import os
from skimage import feature
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import sys
import time
from datetime import datetime as dt

def insert_k_and_dis_func(string):  # insert k and distance function to results.txt
    with open('results.txt', 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(string.rstrip('\r\n') + '\n' + content)

def chi_square(a, b):
    return 0.5 * np.sum((a - b) ** 2 / (a + b + 1e-6))

def change_shape(img):
    WHITE = [255, 255, 255]  # for padding images with incorrect width and height
    height, width = img.shape  # save height and width for current image
    if height > width: # check if the height or width is not in the same value and fix it
        imaPadding= cv2.copyMakeBorder(img, 0, 0, int((height-width)//2), int((height-width)/2), cv2.BORDER_CONSTANT,value=WHITE)
    else:
        imaPadding = cv2.copyMakeBorder(img, int((width-height)//2), int((width-height)/2), 0, 0, cv2.BORDER_CONSTANT,value=WHITE)

    dim = (40, 40)
    resizeImage = cv2.resize(imaPadding, dim) # resize the image

    return resizeImage

def create_cmatrix(testdata_label, y_pred):
    cm = confusion_matrix(testdata_label, y_pred, labels=[f'{i}' for i in range(0, 27)])  # create confusion matrix
    cm_df = pd.DataFrame(data=cm, index=[i for i in range(0, 27)], columns=[i for i in range(0, 27)])
    cm_df.to_csv('confusion_matrix.csv')

    df = pd.DataFrame(data={'Accuracy': (cm.diagonal() / cm.sum(axis=1)) * 100}, columns=['Accuracy'])  # create txt file with accuracy for each letter
    df.index.name = 'Letter'  # rename column name for index
    df.to_csv('results.txt', sep='\t') # save data to text file

now = dt.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

print("Load train set")
startTime = time.time()  # start timer
hogList = list()
hogListLabel = list ()
root_dir = sys.argv[1]  # root directory
train_dir = root_dir + '\\TRAIN' # train directory

for subdir in os.listdir(train_dir):
    for image_name in os.listdir(train_dir + '\\' + subdir):
        image_load = cv2.imread(train_dir + '\\' + subdir + '\\' + image_name, cv2.IMREAD_GRAYSCALE)
        fixedImage = change_shape(image_load)
        ch_hog = feature.hog(fixedImage, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),transform_sqrt=False, block_norm="L2")
        hogList.append(ch_hog)
        hogListLabel.append(subdir)

print("Shuffle and split to train set and validation set")
X_train, X_validate, y_train, y_validate = train_test_split(hogList, hogListLabel, test_size=0.1) # 90% training and 10% test
max = 0
maxPrec1 = tuple()
maxPrec2 = tuple()

print("Start to train the model")

for k in range(1, 16):
    if k % 2 != 0:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean') # Create KNN Classifier
        knn.fit(X_train, y_train) # Train the model using the training sets
        y_pred = knn.predict(X_validate) # Predict the response for test dataset

        if(metrics.accuracy_score(y_validate, y_pred)*100 > max):
            max = metrics.accuracy_score(y_validate, y_pred)*100
            maxPrec1 = (knn, max, k, 'euclidean')

        knn = KNeighborsClassifier(n_neighbors=k, metric=chi_square) # Create KNN Classifier
        knn.fit(X_train, y_train) # Train the model using the training sets
        y_pred = knn.predict(X_validate) # Predict the response for test dataset

        if(metrics.accuracy_score(y_validate, y_pred)*100 > max):
            max = metrics.accuracy_score(y_validate, y_pred)*100
            maxPrec2 = (knn, max, k, 'chi_square')

best_dis_and_k = tuple()
if maxPrec1[1] > maxPrec2[1]:  # checking which function gives the best results
    best_dis_and_k = (maxPrec1[1], maxPrec1[2], maxPrec1[3])
else:
    best_dis_and_k = (maxPrec2[1], maxPrec2[2], maxPrec2[3])

print("Finished to train the model ")
print("Load the test set ")

dir_test = root_dir +'\\TEST'

testdata = list()
testdata_label = list()
for subdir in os.listdir(dir_test):  # load the dataset
    for image_name in os.listdir(dir_test + '\\' + subdir):
        image_load = cv2.imread(dir_test + '\\' + subdir + '\\' + image_name, cv2.IMREAD_GRAYSCALE)
        fixedImage = change_shape(image_load)
        ch_hog = feature.hog(fixedImage, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),transform_sqrt=False, block_norm="L2")
        testdata.append(ch_hog)
        testdata_label.append(subdir)

print("Start final stage with dataset")
if best_dis_and_k[2] == 'chi_square':  # knn evaluation with function that gives the best results
    knn = KNeighborsClassifier(n_neighbors=best_dis_and_k[1], metric=chi_square)  # Create KNN Classifier
else:
    knn = KNeighborsClassifier(n_neighbors=best_dis_and_k[1], metric='euclidean')
knn.fit(hogList, hogListLabel)
y_pred = knn.predict(testdata)


print("Final score for dataset: ", accuracy_score(testdata_label, y_pred)*100)
print("Export data (csv and text file)")

create_cmatrix(testdata_label, y_pred)  # export data to csv
k_and_dis_data = 'k = {0}, distance function is {1}'.format(best_dis_and_k[1], best_dis_and_k[2])
insert_k_and_dis_func(k_and_dis_data)  # add k and dis func to the text file


finish = dt.now()
total_time = finish - now
finish_time = finish.strftime("%H:%M:%S")

print("Finish Time =", finish_time)
print("Done, total time = ", total_time)
