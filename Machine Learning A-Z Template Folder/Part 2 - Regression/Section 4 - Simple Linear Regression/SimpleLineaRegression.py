# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X =StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)"""

# Fitting Simple Linear Regrssion to the Training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
y_pred=regressor.predict(X_test)

# Visulising the Training set results
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_test),color='blue')
plt.title('Salary vs Experiance (Training Set)')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()