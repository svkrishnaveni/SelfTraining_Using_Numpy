#!/usr/bin/env python
'''
This script contains various functions used in this project
Author: Sai Venkata Krishnaveni Devarakonda
Date: 04/26/2022
'''

import numpy as np
import re

########################################## Load data functions################################
# Loading homework1 train data separated as features and targets
def Load_data(str_path_1b_program):
    '''
    This function loads train data(demographic data height,weight,age) from homework1 and separates features and targets
    inputs: str path to train data.txt
    outputs: numpy arrays of targets,features
    '''
    # initialize empty lists to gather features and targets
    features = []
    targets = []
    # read lines in txt file as string
    with open(str_path_1b_program) as f:
        for line in f:
            data = line
            # remove parenthesis
            data_tmp = re.sub(r"[\([{})\]]", "", data)
            # extract list of 1 feature
            lsFeature_tmp = [float(data_tmp.split(',')[0]), float(data_tmp.split(',')[1]), int(data_tmp.split(',')[2])]
            # extract target
            lsTarget_tmp = [data_tmp.split(',')[3][1]]
            features.append(lsFeature_tmp)
            targets.append(lsTarget_tmp)
    features = np.array(features)
    targets = np.array(targets)
    return features,targets

def cartesian_distance(a,b):
    '''
    This function calculates euclidean distance between 2 vectors
    inputs:Two vectors
    outputs:Euclidean Distance between given vectors
    '''
    distance=np.sqrt(np.sum(np.square(a-b)))
    return distance


########################################## Functions for hierarchical clustering ################################
def hierarchical_clustering(features, linkage, clusters_num):
    '''
    This function does hierarchical_clustering based on given linkage
    inputs: features, linkage and number of clusters
    outputs:clusters
    '''
    # calculate distance from every point to every other point in the dataset
    initial_distances = np.zeros([len(features), len(features)])
    for i in range(len(features)):
        for j in range(len(features)):
            initial_distances[i][j] = cartesian_distance(features[i], features[j])
    # making all the diagonal elements from 0 to inf to make easier to get min value using min function
    np.fill_diagonal(initial_distances, np.inf)
    clusters = get_clusters(initial_distances, linkage)

    iteration_number = initial_distances.shape[0] - clusters_num
    final_clusters = clusters[iteration_number]
    unique_final_clusters = np.unique(final_clusters)
    return final_clusters, unique_final_clusters


def get_clusters(initial_distances, linkage):
    '''
    This function gets clusters
    inputs: inital distances between data points, linkage
    outputs: clusters with datapoint indices
    '''
    clusters = {}
    row_index = -1
    col_index = -1
    arr = []

    # Make every datapoint as one cluster.Consider only indices
    for n in range(initial_distances.shape[0]):
        arr.append(n)

    clusters[0] = arr.copy()

    # find minimum value from the distance matrix
    for k in range(1, initial_distances.shape[0]):
        min_val = np.inf

        for i in range(0, initial_distances.shape[0]):
            for j in range(0, initial_distances.shape[1]):
                if (initial_distances[i][j] <= min_val):
                    min_val = initial_distances[i][j]
                    row_index = i
                    col_index = j

        # for Single Linkage
        if (linkage == "single"):
            for i in range(0, initial_distances.shape[0]):
                if (i != col_index):
                    temp = min(initial_distances[col_index][i], initial_distances[row_index][i])
                    initial_distances[col_index][i] = temp
                    initial_distances[i][col_index] = temp

        # for Complete Linkage
        elif (linkage == "complete"):
            for i in range(0, initial_distances.shape[0]):
                if (i != col_index and i != row_index):
                    temp = max(initial_distances[col_index][i], initial_distances[row_index][i])
                    initial_distances[col_index][i] = temp
                    initial_distances[i][col_index] = temp

        # for Average Linkage
        elif (linkage == "average"):
            for i in range(0, initial_distances.shape[0]):
                if (i != col_index and i != row_index):
                    temp = (initial_distances[col_index][i] + initial_distances[row_index][i]) / 2
                    initial_distances[col_index][i] = temp
                    initial_distances[i][col_index] = temp

        # set the rows and columns for the cluster with higher index i.e. the row index to infinity
        for i in range(0, initial_distances.shape[0]):
            initial_distances[row_index][i] = np.inf
            initial_distances[i][row_index] = np.inf

        minimum = min(row_index, col_index)
        maximum = max(row_index, col_index)
        for n in range(len(arr)):
            if (arr[n] == maximum):
                arr[n] = minimum
        clusters[k] = arr.copy()

    return clusters

def most_common_label(labels):
    '''
      This function finds the most common label
      Input : labels
      Output : most common label
      '''
    vals, counts = np.unique(labels, return_counts=True)
    mode_value = np.argwhere(counts == np.max(counts))
    most_frequent_label = vals[mode_value][0][0]
    return most_frequent_label

########################################## Functions for self training ################################
def knn_classifier(train_data,train_labels,sample,k):
    '''
    This function is KNN classifier
    inputs:
        train_features = mxn array (m= #observations,n= #features)
        train_labels = nx1 array of targets
        sample = 1 row of test features
        k =  int (#nearest neighbors)
    outputs:
        Euclidean distance of k neighbors
        Targets of k nearest neighbors
    '''
    distance=[]
    neighbors=[]
    for i in range(len(train_data)):
        d=cartesian_distance(train_data[i],sample)
        distance.append(d)
    ind = np.argsort(distance)
    distance.sort()
    for i in range(k):
        neighbors.append(distance[i])
    targets = [train_labels[v] for v in ind[:k]]
    return neighbors,targets

def accuracy(y_true, y_pred):
    '''
    This function calculates accuracy
    Input : actual and expected labels
    Output : accuracy
    '''
    count = 0
    for i in range(len(y_true)):
        if (y_true[i] == y_pred[i]):
            count = count + 1
    accuracy = count / len(y_true) * 100
    return accuracy


def self_training(s_train_features, s_train_targets, us_train_features, num, k):
    '''
    This function self-training using a KNN classifier
    Input : array of supervised data(features,targets) and unsupervised data(features), number of data points to be added in each iteration,k
    Output : array of unsupervised data predicted labels
    '''
    tmp_s_train_features = s_train_features
    tmp_s_train_targets = s_train_targets
    tmp_s_train_targets = tmp_s_train_targets.ravel()
    tmp_us_train_features = us_train_features

    while len(tmp_us_train_features) != 0:
        target = []
        for z in range(num):
            sample = tmp_us_train_features[z]
            neighbors, t = knn_classifier(tmp_s_train_features, tmp_s_train_targets, sample, k)
            values, counts = np.unique(t, return_counts=True)
            ind = np.argmax(counts)
            pred_t = values[ind]
            target.append(pred_t)
        adding_data_elements = tmp_us_train_features[0:num]
        tmp_s_train_features = np.append(tmp_s_train_features, adding_data_elements, axis=0)
        tmp_s_train_targets = np.append(tmp_s_train_targets, target, axis=0)
        tmp_us_train_features = np.delete(tmp_us_train_features, slice(0, num), axis=0)
    return tmp_s_train_targets