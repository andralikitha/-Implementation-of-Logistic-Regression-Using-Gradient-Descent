# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.


## Program:
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X = data[:, [0,1]]
y = data[:, 2]

print("Array value of X:")
X[:5]

print("Array value of Y:")
y[:5]

print("Exam 1-score graph:")
plt.figure()
plt.scatter(X[y==1][:, 0],X[y==1][:, 1], label="Admitted")
plt.scatter(X[y==0][:, 0],X[y==0][:, 1], label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))
  
print("Sigmoid function graph:")
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J = -(np.dot(y,np.log(h)) + np.dot(1-y, np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y)/ X.shape[0]
  return J,grad
  
print("X_train_grad value:")
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

print("Y_train_grad value:")
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J = -(np.dot(y, np.log(h)) + np.dot(1-y, np.log(1-h))) / X.shape[0]
  return J
  
def gradient(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y) / X.shape[0]
  return grad 
  
print("Print res.x:")
X_train = np.hstack((np.ones((X.shape[0], 1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train,y), method='Newton-CG', jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max= X[:,0].min()-1, X[:,0].max()+1
  y_min, y_max= X[:,0].min()-1, X[:,0].max()+1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted") 
  plt.contour(xx, yy, y_plot, levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
print("Decision boundary-graph for exam score:")
plotDecisionBoundary(res.x,X,y)

print("Probability value:")
prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)
  
print("Prediction value of mean:")
np.mean(predict(res.x,X) == y)


/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: ANDRA LIKITHA
RegisterNumber: 212221220006 
*/
```

## Output:
![234284153-2b1b8eaa-de3b-490f-bce0-dbc8afd47bb0](https://user-images.githubusercontent.com/131592130/236865934-924989ad-8b43-4565-8c3e-071f61e73228.png)
![234284226-e6bf131d-2cad-4670-99d7-fb51f14cc540](https://user-images.githubusercontent.com/131592130/236866054-7726430e-d8dd-407d-a4d1-66b32efda6c4.png)
![234284284-ad592394-0173-48ef-9a07-ee49ed52dda5](https://user-images.githubusercontent.com/131592130/236866146-5261ed5c-b62e-4c05-830a-dbe56211b8f0.png)
![234284343-76f92d47-8058-43a0-ad3c-25564702691e](https://user-images.githubusercontent.com/131592130/236866252-f6a9c9b5-4016-46b8-8dd1-a3cd05f23205.png)
![234284400-7ff5576d-212b-4c49-9474-6e435f488b47](https://user-images.githubusercontent.com/131592130/236866394-e62411b6-7abd-40a3-9003-18ea61b595c2.png)
![234284475-102620b9-7ded-4933-87eb-40ab97381e12](https://user-images.githubusercontent.com/131592130/236866490-17578189-a087-459b-bdb1-9ee311d6022c.png)
![234284572-331fef27-5f30-44d0-a0f3-75ecc27234f7](https://user-images.githubusercontent.com/131592130/236866590-9273d769-e785-4c49-8664-9a548654a6e3.png)
![234284634-b1a90d56-34ed-4d15-943a-8431aea96ad3](https://user-images.githubusercontent.com/131592130/236866703-f40cac48-a9b0-43f2-a985-653d7e079f44.png)
![234284681-6b4a65c8-2b7a-4c6f-86df-22a23bcc1fb1](https://user-images.githubusercontent.com/131592130/236866894-f223c95d-44a5-47e9-95de-63341a507a02.png)
![234286219-fdbd9598-a1b8-4d5d-9e12-d4d4f12e1b89](https://user-images.githubusercontent.com/131592130/236867003-04feb781-815b-417f-907a-8083e412afe5.png)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

