# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# IN OUR CASE:AS THE NUMBER OF TRAINING SAMPLES FOR EACH CASE IS EQUAL

import math
import numpy as np
import pandas as pd
import scipy as sp
import time
import os
from math import sqrt
from math import pi
from math import exp
import sys

def main(trainfile, testfile):

    start = time.time()

    # Press the green button in the gutter to run the script.
    #if __name__ == '__main__':
        #print_hi('Welcome to MP1')

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

    #testfile = './testing.txt'
    #trainfile = './training.txt'
    traindata = np.loadtxt(trainfile, delimiter=',', dtype=str)
    testdata = np.loadtxt(testfile, delimiter=',', dtype=str)
    rows = np.size(traindata, 0)
    columns = np.size(traindata, 1)

    #print(traindata[0])
    for i in range(rows):

        if traindata[i][1] == 'M':
            traindata[i][1] = '0'
        elif traindata[i][1] == 'F':
            traindata[i][1] = '1'

    for i in range(np.size(testdata, 0)):
        if testdata[i][1] == 'M':
            testdata[i][1] = '0'
        elif testdata[i][1] == 'F':
            testdata[i][1] = '1'

    np.savetxt('traindata.txt', traindata, delimiter=',', fmt="%s")
    traindata = np.loadtxt('traindata.txt', delimiter=',')

    np.savetxt('testdata.txt', testdata, delimiter=',', fmt="%s")
    testdata = np.loadtxt('testdata.txt', delimiter=',')


    rows = np.size(traindata, 0)
    columns = np.size(traindata, 1)

    #print('------------------- Train Data --------------------- \n', traindata)
    #print('\n------------------- Test Data ---------------------- \n', testdata)
    #
    # Example of separating data by class value
    #
    # print(rows)
    # print(columns)

    # CREATING DICTIONARY CORRESPONDING TO PERFORMANCE CLASSES (KEY)
    data_classes = dict()
    for i in range(rows):
        tmp = traindata[i][columns - 1]
        if tmp not in data_classes:
            new_class = tmp
            data_classes[tmp] = list()
        data_classes[tmp].append(traindata[i][0:columns])

    # CHECKING ELEMENTS FOR EACH CLASS
   # for key in data_classes:
        #print("The key is ", key)
        #print(" Number of elements for this key is ", len(data_classes.get(key)))
        # for row in data_classes[label]:
        #     print(row)

    # Calculate the average of each column for every label (High performance or Low Performance)
    gauss_stat_features = [[], []]
    ctr = 0
    for key in data_classes:
        # print("Gathering statistical data for trait and class correlation for key:", key)
        a = np.array(data_classes[key])
        mean_for_features = a.mean(axis=0)
        mean_for_features = np.delete(mean_for_features, [columns - 1])
        std_for_features = a.std(axis=0)
        std_for_features = np.delete(std_for_features, [columns - 1])

        # print("\n The mean for each feature for this key is :", mean_for_features)
        # print("\n The standard deviation for each feature for this key is :", std_for_features)

        # gauss_stat_features[ctr] = list()
        gauss_stat_features[ctr] = (np.array(np.transpose(np.array(
            [mean_for_features, std_for_features, len(data_classes.get(key)) * np.ones(columns - 1, dtype=int)]))).tolist())
        ctr = ctr + 1

    #print("\n Gaussian statistical data ", gauss_stat_features)
    total_rows = np.size(traindata, 0)

    Training_time = time.time()
    #print("\n Training Time took", Training_time - start, "seconds")
    # Calculating class probability for different classes

    inputdata = traindata
    # print("The input data is \n", inputdata)
    # print("The length of columns of inputdata is \n", np.size(inputdata,1 ))
    Data_classification = []
    classify = []

    start = time.time()
    for j in range(np.size(traindata, 0)):  # DATA --> training

        x = inputdata[j][0:columns - 1]
        # print(" The testing input is", x)
        class_predict_prob = []
        for i in range(len(gauss_stat_features)):  # Number of Classes to be considered
            P_class = gauss_stat_features[i][0][2] / total_rows
            temp = math.log(P_class)
            # print("\n \n The P_class value for the class", i, "is", temp)
            for feature in range(columns - 1):
                mean = np.array(gauss_stat_features[i][feature][0])
                stdev = np.array(gauss_stat_features[i][feature][1])
                # print("Mean for attribute", feature + 1, "is", mean)
                # print("Standard Deviation for attribute", feature + 1, "is", stdev)

                term1 = -(((x[feature] - mean)/stdev) ** 2) / 2
                temp = temp + (math.log(1 / stdev)) + term1
                # print("The temp value after feature", feature + 1, "is", temp)
            class_predict_prob.append(temp)
        # print("\n \n The class probabilities for the input at index", j, "are", class_predict_prob)
        Data_classification.append(class_predict_prob)
        # print("\n \n The class probabilities for all inputs are", Data_classification)

        if class_predict_prob[0] > class_predict_prob[1]:
            classify.append(0)
        elif class_predict_prob[1] > class_predict_prob[0]:
            classify.append(1)

    classify = np.array(classify)
    end = time.time()
    #print("\n Checking Model Accuracy on training data took", end - start, "seconds")

    #print(classify)
    Accuracy_vector = []
    hit_count_training = 0
    for i in range(np.size(traindata, 0)):

        if traindata[i][11] == classify[i]:
            Accuracy_vector.append(1)
            hit_count_training = hit_count_training + 1
        else:
            Accuracy_vector.append(0)


    #print("Number of train data is", np.size(traindata, 0))
    #print("\n The accuracy vector for train dataset is", Accuracy_vector)
    #print("\n Hit accuracy achieved for train dataset is:",100*hit_count_training/np.size(traindata, 0))


    inputdata = testdata
    # print("The input data is \n", inputdata)
    # print("The length of columns of inputdata is \n", np.size(inputdata,1 ))
    Data_classification = []
    classify = []

    start = time.time()
    for j in range(np.size(testdata, 0)):  # DATA --> test

        x = inputdata[j][0:columns - 1]
        # print(" The testing input is", x)
        class_predict_prob = []
        for i in range(len(gauss_stat_features)):  # Number of Classes to be considered
            P_class = gauss_stat_features[i][0][2] / total_rows
            temp = math.log(P_class)
            # print("\n \n The P_class value for the class", i, "is", temp)
            for feature in range(columns - 1):
                mean = np.array(gauss_stat_features[i][feature][0])
                stdev = np.array(gauss_stat_features[i][feature][1])
                # print("Mean for attribute", feature + 1, "is", mean)
                # print("Standard Deviation for attribute", feature + 1, "is", stdev)

                term1 = -(((x[feature] - mean)/stdev) ** 2) / 2
                temp = temp + (math.log(1 / stdev)) + term1
                # print("The temp value after feature", feature + 1, "is", temp)
            class_predict_prob.append(temp)
        # print("\n \n The class probabilities for the input at index", j, "are", class_predict_prob)
        Data_classification.append(class_predict_prob)
        # print("\n \n The class probabilities for all inputs are", Data_classification)

        if class_predict_prob[0] > class_predict_prob[1]:
            classify.append(0)
            print("0")
        elif class_predict_prob[1] > class_predict_prob[0]:
            classify.append(1)
            print("1")

    classify = np.array(classify)
    end = time.time()
    #print("\n Checking Model Accuracy on test dataset took", end-start, "seconds")

    #print(classify)
  #  Accuracy_vector = []
  #   hit_count = 0
  #   for i in range(np.size(testdata, 0)):
  #
  #      if testdata[i][11] == classify[i]:
  #          Accuracy_vector.append(1)
  #          hit_count = hit_count + 1
  #      else:
  #         Accuracy_vector.append(0)
           


    #print("Number of test data is", np.size(testdata, 0))
    #print("\n The accuracy vector for test dataset is", Accuracy_vector)
    #print("\n Hit accuracy achieved for test dataset is:", 100*hit_count/np.size(testdata, 0))



if __name__ == "__main__":
    #print('HEY')
    #print(sys.argv)
    main(sys.argv[1],sys.argv[2])
