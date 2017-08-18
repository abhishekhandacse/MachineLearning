# Data Preprocessing

#importing the dataset

dataset=read.csv('50_Startups.csv')
dataset=dataset[,2:3]
# Spliting the dataset into the Training set and Test set
#install.packages('caTools')
library(caTools)
set.seed(123)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)

#  #Feature Scaling
#  training_set[,2:3]=scale(training_set[,2:3])
#  test_set[,2:3]=scale(test_set[,2:3])

