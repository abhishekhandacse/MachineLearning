# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem ( word ) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features = 1500)
X=cv.fit_transform(corpus).toarray()
Y=dataset.iloc[:,1].values

# Using Naive base model we obtained
 
#Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
#
## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X =StandardScaler()
#X_train=sc_X.fit_transform(X_train)
#X_test=sc_X.transform(X_test)
#
## Fitting Logistic Regression to the Training set
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(X_train,Y_train)
## Predicting the Test set results
#y_pred=classifier.predict(X_test)
#y_pred
## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(Y_test,y_pred,)
#
# Now using Decision Tree
#Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
#
## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X =StandardScaler()
#X_train=sc_X.fit_transform(X_train)
#X_test=sc_X.transform(X_test)
#
## Fitting Logistic Regression to the Training set
#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
#classifier.fit(X_train,Y_train)
## Predicting the Test set results
#y_pred=classifier.predict(X_test)
#y_pred
## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(Y_test,y_pred,)
#cm

##Random Forest Classification
#from sklearn.cross_validation import train_test_split
#X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
#
## Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X =StandardScaler()
#X_train=sc_X.fit_transform(X_train)
#X_test=sc_X.transform(X_test)
#
## Fitting Logistic Regression to the Training set
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators=1000,criterion= 'entropy',random_state=0)
#classifier.fit(X_train,Y_train)
## Predicting the Test set results
#y_pred=classifier.predict(X_test)
#y_pred
## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(Y_test,y_pred,)
#cm

# In Logistic Regression actually Maximum entropy is implicily implemented
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)