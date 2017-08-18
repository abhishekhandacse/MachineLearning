# Decision Tree Regression


#importing the dataset

dataset=read.csv('Position_Salaries.csv')
dataset=dataset[,2:3]

# Encoding categorical data
#dataset$State=factor(dataset$State,
#                    levels=c('New York','California','Florida'),
#                   labels=c(1,2,3))

# Spliting the dataset into the Training set and Test set
#install.packages('caTools')
#library(caTools)
#set.seed(123)
#training_set=subset(dataset,split==TRUE)
#test_set=subset(dataset,split==FALSE)

#  #Feature Scaling
#  training_set[,2:3]=scale(training_set[,2:3])
#  test_set[,2:3]=scale(test_set[,2:3])

#Fitting Polynomial Regression to the dataset
#Create your regressor here
#Fitting the Tree Regression to the dataset
# install.packages('rpart')
library(rpart)
regressor =rpart(formula = Salary ~ .,data = dataset,control = rpart.control(minsplit = 1))

# Predicting a new Result 
y_pred = predict(regressor,data.frame(Level = 6.5))

#For Higher Resolution graph and smoother curve
library(ggplot2)
x_grid=seq(min(dataset$Level),max(dataset$Level),0.01)

ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             colour='red')+
  geom_line(aes(x=x_grid,y=predict(regressor,newdata = data.frame(Level=x_grid))),
            colour='blue')+
  ggtitle('Truth or Bluff Decision Tree Regression')+
  xlab('Level') +
  ylab('Salary')

