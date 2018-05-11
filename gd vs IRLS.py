#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 01:30:00 2018

@author: shashi
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('kc_house_data.csv') 

len_dataset = len(dataset.index) #num of elements in the dataset

len_test = (int)(len_dataset*0.2)
len_train = len_dataset-len_test

train = pd.DataFrame(data=dataset)
train.drop(train.index[0:], inplace=True)

l1 = random.sample(range(0,len_dataset), len_train)
l2 =[]
for i in  range(len_dataset):
    l2.append(i)
    
l3 = (list(set(l2) - set(l1)))
#train = dataset.iloc[l1]
#test = dataset.iloc[l3]
train = dataset.iloc[:len_train]
test = dataset.iloc[len_train:]


train_mean = train.mean()
train_std = train.std()

#normalized training data
train = (train -train_mean)/train_std


#splitting the training data into features and output
train_x = train.iloc[:,0:4]
train_y = train.iloc[:,4]

#splitting the test data into features and output
test_x = test.iloc[:,0:4]
test_y = test.iloc[:,4]

#normalizing the test data feature space
test_x = (test_x-test_x.mean())/test_x.std()

#Converting train features into a numpy matrix
x = train_x.as_matrix()

#Inserting a column of ones in the beginning of training features
x = np.concatenate((np.ones(len_train)[:, np.newaxis], x), axis=1)


#extracting the outputs values of training set
y = train_y.as_matrix()

#converting the test feature space into a numpy matrix
x_test = test_x.as_matrix()

#Inserting a column of ones in front of x_test
x_test = np.concatenate((np.ones(len_test)[:, np.newaxis], x_test), axis=1)

y_test = test_y.as_matrix()
theta = np.array([0,0,0,0,0])

#the learning rate
alpha = 0.05

#numpy array for rmse values over the iterations 
rmse = np.zeros((500,))


temp = np.subtract((np.matmul(theta.transpose(),x_test.transpose())*train_std[4] + train_mean[4]),y_test.transpose())
i=0
sum =0
for i in range(temp.size):
    sum = sum + temp[i]**2
rmse[0] = (sum/temp.size)**(1/2)
    
leng = 2*len_test
j=0
for j in range(1,500):
    theta = theta - (alpha/leng)*((np.matmul((np.subtract(np.matmul(x,theta),y)).transpose(),x)).transpose())
    res = np.matmul(theta.transpose(),x_test.transpose())*train_std[4] + train_mean[4]
    temp = np.subtract(res,y_test.transpose())
    i=0
    sum =0
    for i in range(temp.size):
        sum = sum + temp[i]**2
    rmse[j] = (sum/temp.size)**(0.5)

print('Value of Learned Parameters in case of  Gradient descent')
i=0
for i in range(5):
    print('theta '+str(i)+' : '+str(theta[i]))

rmse_irl = np.zeros((500,))
j=0 
theta_irl =  np.matmul((np.matmul((np.linalg.inv(np.matmul(x.transpose(),x))),x.transpose())),y)  
for j in range(500):
    res = np.matmul(theta_irl.transpose(),x_test.transpose())*train_std[4] + train_mean[4]
    temp = np.subtract(res,y_test.transpose())
    i=0
    sum =0
    for i in range(temp.size):
        sum = sum + temp[i]**2
    rmse_irl[j] = (sum/temp.size)**(0.5)
    
print()
print()
print()
print()
print('Value of Learned Parameters in case of IRLS')
i=0
for i in range(5):
    print('theta '+str(i)+' : '+str(theta_irl[i]))
    
x_axis = np.linspace(1, 500, 500)
plt.title('RMSE vs ITERATIONS')
plt.plot(x_axis, rmse,'r', label='gradient descent')
plt.plot(x_axis, rmse_irl,'g', label='IRLS')
plt.xlabel('iteration')
plt.ylabel('rmse')
plt.grid() # grid on
plt.legend() 
plt.show()

