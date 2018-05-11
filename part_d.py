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

leng = 2*len_test


theta = np.array([0,0,0,0,0])

alpha_common  = np.linspace(0.001, 0.05, 50)


rmse_linear = np.zeros(50)
k=0
for k in range(50):
    theta = np.zeros(5,)
    j=0
    for j in range(500):
        theta = theta - (alpha_common[k]/leng)*((np.matmul((np.subtract(np.matmul(x,theta),y)).transpose(),x)).transpose())
        
    res = np.matmul(theta.transpose(),x_test.transpose())*train_std[4] + train_mean[4]
    temp = np.subtract(res,y_test.transpose())
    i=0
    sum =0
    for i in range(temp.size):
        sum = sum + temp[i]**2
    rmse_linear[k] = (sum/temp.size)**(0.5)
    
    

print('Value of Learned Parameters in case of Mean Squared Error Function')
i=0
for i in range(5):
            print('theta '+str(i)+' : '+str(theta[i]))  
    
    
k=0
rmse_mean_cost = np.zeros(50)
for k in range(50):
    theta_abs_cost = np.zeros(5)
    #j=0
    #res = np.matmul(theta_abs_cost.transpose(),x_test.transpose())*train_std[4] + train_mean[4]
    #tempp = np.subtract(res,y_test.transpose())
    ##i=0
    #sum =0
    #for i in range(tempp.size):
        #sum = sum + tempp[i]**2
    #rmse_mean_cost[k][0] = (sum/tempp.size)**(0.5)  
    for j in range(1,100):
        temp = (np.subtract(np.matmul(x,theta_abs_cost),y))
        ones_array = np.ones(temp.size)
        l=0
        for l in range(temp.size):
            if(temp[l] > 0) :
                ones_array[l] = 1
            else :
                ones_array[l] = -1
        theta_abs_cost = theta_abs_cost - (alpha_common[k]/leng)*((np.matmul(ones_array.transpose(),x)).transpose())
    res = np.matmul(theta_abs_cost.transpose(),x_test.transpose())*train_std[4] + train_mean[4]
    tempp = np.subtract(res,y_test.transpose())
    i=0
    sum =0
    for i in range(tempp.size):
        sum = sum + tempp[i]**2
    rmse_mean_cost[k] = (sum/tempp.size)**(0.5)
    
    
print()
print()
print()
print() 
print('Value of Learned Parameters in case of Mean Absolute Error Function')
i=0
for i in range(5):
            print('theta '+str(i)+' : '+str(theta_abs_cost[i]))   
    
    
k=0
rmse_cubic_cost = np.zeros(50)
for k in range(50):
    j=0
    theta_cubic_cost = np.zeros(5)
    for j in range(1,80):
        tempp = (np.subtract(np.matmul(x,theta_cubic_cost),y))**2
        theta_cubic_cost = theta_cubic_cost - 3*(alpha_common[0]/leng)*((np.matmul(tempp.transpose(),x)).transpose())
    
    res = np.matmul(theta_cubic_cost.transpose(),x_test.transpose())*train_std[4] + train_mean[4]
    temp = np.subtract(res,y_test.transpose())
    i=0
    sum =0
    for i in range(temp.size):
        sum = sum + temp[i]**2
    rmse_cubic_cost[k] = (sum/temp.size)**(0.5)



print()
print()
print()
print() 
print('Value of Learned Parameters in case of Mean cube error function')
i=0
for i in range(5):
            print('theta '+str(i)+' : '+str(theta_cubic_cost[i]))

plt.title('RMSE vs Learning Rate')
plt.plot(alpha_common, rmse_mean_cost,'r', label='Mean Absolute Error')
plt.plot(alpha_common, rmse_linear,'b', label='Mean Square Error')
plt.ylabel('RMSE')
plt.xlabel('Learning Rate')
plt.grid() # grid on
plt.legend() 
plt.show() 


plt.title('RMSE vs Learning Rate')
plt.plot(alpha_common, rmse_cubic_cost,'g', label='Mean Cube Error')
plt.ylabel('RMSE')
plt.xlabel('Learning Rate')
plt.grid() # grid on
plt.legend() 
plt.show()     