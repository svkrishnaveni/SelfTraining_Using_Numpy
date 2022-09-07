#!/usr/bin/env python
'''
This script applies hierarchical clustering implementation with complete linkage to the data.
Author: Sai Venkata Krishnaveni Devarakonda
Date: 04/30/2022
'''

from utilities import Load_data,self_training,accuracy
str_path_1b_program = './data_2c2d3c3d_program.txt'

#segregating the given data into supervised and unsupervised data
arr_features, arr_targets = Load_data(str_path_1b_program)
s_train_features = arr_features[0:20]
s_train_targets = arr_targets[0:20]
us_train_features = arr_features[20:]
us_train_targets = arr_targets[20:]

#data points to be added in each iteration
c = [1,5,100,25]

for c in c:
    tmp_s_train_targets = self_training(s_train_features,s_train_targets,us_train_features,c,5)
    #accuracy for initially unlabelled data
    init_unlabeled_data_pred = tmp_s_train_targets[20:]
    acc = accuracy(us_train_targets, init_unlabeled_data_pred)

    print('Accuracy for initially unlabeled data using self learning(KNN classifier) by adding  ' +str(c)+' data points in each iteration is ' +str(acc))