#Polynomial Regression

#importing the dataset

dataset=read.csv('Position_Salaries.csv')
dataset=dataset[,2:3]

# Encoding categorical data
dataset$State=factor(dataset$State,
                     levels=c('New York','California','Florida'),
                     labels=c(1,2,3))

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

# Predicting a new Result 
y_pred = predict(regressor,data.frame(Level = 6.5))


# Visualizing the  Regression results

library(ggplot2)
ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             colour='red')+
  geom_line(aes(x=dataset$Level,y=predict(regressor,newdata = dataset)),
            colour='blue')+
  ggtitle('Truth or Bluff (Polynomial Regression)')+
  xlab('Level') +
  ylab('Salary')

# Visualizing the  Regression results

#For Higher Resolution graph and smoother curve
x_grid=seq(min(dataset$Level),max(dataset$Level),0.1)
library(ggplot2)
ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             colour='red')+
  geom_line(aes(x=dataset$Level,y=predict(regressor,newdata = data.frame(Level=x_grid))),
            colour='blue')+
  ggtitle('Truth or Bluff (Polynomial Regression)')+
  xlab('Level') +
  ylab('Salary')

