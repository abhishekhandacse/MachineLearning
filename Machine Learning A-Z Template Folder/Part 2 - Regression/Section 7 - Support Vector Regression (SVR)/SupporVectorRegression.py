#Polynomial Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

# Encoding categorical data
# Encoding the Independent variable
"""from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X=X[:,1:]

#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
"""
# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X =StandardScaler()
sc_Y =StandardScaler()
X=sc_X.fit_transform(X)
Y=sc_Y.fit_transform(Y)

#Fittiong the SVR to the dataset(Does not include feature scaling)
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,Y)

# Predicting a new result with polnomial Regression
y_pred=sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array( [[6.5 ]] ) )))
y_pred

#y_pred=regressor.predict(6.5)
# Visualizing the linear Regression results
plt.scatter(X , Y , color = 'red')
plt.plot(X,regressor.predict(X),color = 'blue')
plt.title('Truth or Bluff( SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualizing the linear Regression results(For High Resolution)
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid) ,1))
plt.scatter(X , Y , color = 'red')
plt.plot(X,regressor.predict(),color = 'blue')
plt.title('Truth or Bluff( Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

