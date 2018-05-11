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
    
x_axis = np.linspace(1, 500, 500)
plt.title('RMSE vs ITERATIONS')
plt.plot(x_axis, rmse,'r', label='gradient descent')
plt.plot(x_axis, rmse_irl,'g', label='IRLS')
plt.xlabel('iteration')
plt.ylabel('rmse')
plt.grid() # grid on
plt.legend() 
plt.show()




rmse_reg = np.zeros((50,))
lamb  = np.linspace(0.2, 10, 50)


for k in range(0,50):
    theta_reg = np.array([0,0,0,0,0])
    for j in range(1,500):
        temp = theta_reg - (alpha/leng)*((np.matmul((np.subtract(np.matmul(x,theta_reg),y)).transpose(),x)).transpose())
        theta_reg = temp - theta_reg*alpha*lamb[k]/leng
        theta_reg[0] = temp[0]
        
    res = np.matmul(theta_reg.transpose(),x_test.transpose())*train_std[4] + train_mean[4]
    temp = np.subtract(res,y_test.transpose())
    i=0
    sum =0
    for i in range(temp.size):
        sum = sum + temp[i]**2
    rmse_reg[k] = (sum/temp.size)**(0.5)
    
plt.title('RMSE vs Weight of Regularisation')
plt.plot(lamb, rmse_reg)
plt.ylabel('RMSE')
plt.xlabel('lambda')
plt.grid() # grid on
plt.legend() 
plt.show()
 

    


dataset_quad = dataset.copy(deep = True)
for i in range(0,4):
    for j in range(i,4):
        tempo = (dataset_quad.iloc[:,i].values)*(dataset_quad.iloc[:,j].values)  
        s= 'x' + str(i)+ 'x' +str(j)
        dataset_quad.loc[:,s] = pd.Series(tempo, index=dataset_quad.index)
print(dataset_quad.shape)
dataset_cubic = dataset_quad.copy(deep = True)
for i in range(0,4):
    for j in range(i,4):
        for k in range(j,4):
            tempo = (dataset.iloc[:,i].values)*(dataset.iloc[:,j].values)*(dataset.iloc[:,k].values) 
            s= 'x' + str(i) + 'x' +str(j) + 'x' + str(k)
            dataset_cubic.loc[:,s] = pd.Series(tempo, index=dataset.index)
print(dataset_cubic.shape)        
     
train_quad = dataset_quad.iloc[:len_train]
train_quad_mean = train_quad.mean()
train_quad_std = train_quad.std()
train_quad = (train_quad -train_quad_mean)/train_quad_std      

train_quad_x = train_quad.iloc[:, train_quad.columns !='price']
train_quad_x = train_quad_x.as_matrix()
train_quad_x = np.concatenate((np.ones(len_train)[:, np.newaxis], train_quad_x), axis=1)
train_quad_y = train_quad.iloc[:,4].as_matrix()

test_quad = dataset_quad.iloc[len_train:]
test_quad_x = test_quad.iloc[:, test_quad.columns !='price']
test_quad_y = test_quad.iloc[:,4].as_matrix()
test_quad_x = (test_quad_x - (test_quad_x.mean()))/test_quad_x.std()
test_quad_x = test_quad_x.as_matrix()
test_quad_x = np.concatenate((np.ones(len_test)[:, np.newaxis], test_quad_x), axis=1)



train_cubic = dataset_cubic.iloc[:len_train]
train_cubic_mean = train_cubic.mean()
train_cubic_std = train_cubic.std()
train_cubic = (train_cubic -train_cubic_mean)/train_cubic_std      

train_cubic_x = train_cubic.iloc[:, train_cubic.columns !='price']
train_cubic_x = train_cubic_x.as_matrix()
train_cubic_x = np.concatenate((np.ones(len_train)[:, np.newaxis], train_cubic_x), axis=1)
train_cubic_y = train_cubic.iloc[:,4].as_matrix()

test_cubic = dataset_cubic.iloc[len_train:]
test_cubic_x = test_cubic.iloc[:, test_cubic.columns !='price']
test_cubic_y = test_cubic.iloc[:,4].as_matrix()
test_cubic_x = (test_cubic_x - (test_cubic_x.mean()))/test_cubic_x.std()
test_cubic_x = test_cubic_x.as_matrix()
test_cubic_x = np.concatenate((np.ones(len_test)[:, np.newaxis], test_cubic_x), axis=1)



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





rmse_quad = np.zeros(50)
k=0
for k in range(50):
    theta_quad = np.zeros(train_quad_x.shape[1],)
    j=0
    for j in range(1,500):
        theta_quad = theta_quad - (alpha_common[k]/leng)*((np.matmul((np.subtract(np.matmul(train_quad_x,theta_quad),train_quad_y)).transpose(),train_quad_x)).transpose())
    res = np.matmul(theta_quad.transpose(),test_quad_x.transpose())*train_quad_std[4] + train_quad_mean[4]
    temp = np.subtract(res, test_quad_y.transpose())
    i=0
    sum =0
    for i in range(temp.size):
        sum = sum + temp[i]**2
    rmse_quad[k] = (sum/temp.size)**(0.5)





rmse_cubic = np.zeros(50)
k=0
for k in range(50):
    theta_cubic = np.zeros(train_cubic_x.shape[1],)
    j=0
    for j in range(1,500):
        theta_cubic = theta_cubic - (alpha_common[k]/leng)*((np.matmul((np.subtract(np.matmul(train_cubic_x,theta_cubic),train_cubic_y)).transpose(),train_cubic_x)).transpose())
    res = np.matmul(theta_cubic.transpose(),test_cubic_x.transpose())*train_cubic_std[4] + train_cubic_mean[4]
    temp = np.subtract(res, test_cubic_y.transpose())
    i=0
    sum =0
    for i in range(temp.size):
        sum = sum + temp[i]**2
    rmse_cubic[k] = (sum/temp.size)**(0.5)
     
 
    
plt.title('RMSE vs Learning Rate')
plt.plot(alpha_common, rmse_linear,'r', label='linear')
plt.plot(alpha_common, rmse_quad,'b', label='quadratic')
plt.plot(alpha_common, rmse_cubic,'g', label='Cubic')
plt.ylabel('RMSE')
plt.xlabel('Learning Rate')
plt.grid() # grid on
plt.legend() 
plt.show()    
    
    
   

 
    

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
        
 

plt.title('RMSE vs Learning Rate')
plt.plot(alpha_common, rmse_mean_cost,'r', label='Mean Absolute Error')
plt.plot(alpha_common, rmse_linear,'b', label='Mean Square Error')
plt.plot(alpha_common, rmse_cubic_cost,'g', label='Mean Cube Error')
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
        
