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

#Fitting Linear Regression to the dataset

lin_reg=lm(formula = Salary ~ .,data = dataset)

#Fitting Polynomial Regression to the dataset

dataset$Level2=dataset$Level^2
dataset$Level3=dataset$Level^3
dataset$Level4=dataset$Level^4
poly_reg = lm(formula = Salary ~ .,data = dataset)

# Visualizing the Linear Regression results
library(ggplot2)
ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             colour='red')+
  geom_line(aes(x=dataset$Level,y=predict(lin_reg,newdata = dataset)),
            colour='blue')+
  ggtitle('Truth or Bluff (Linear Regression)')+
  xlab('Level') +
  ylab('Salary')
# Visualizing the Polynomial Regression results

library(ggplot2)
ggplot()+
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             colour='red')+
  geom_line(aes(x=dataset$Level,y=predict(poly_reg,newdata = dataset)),
            colour='blue')+
  ggtitle('Truth or Bluff (Polynomial Regression)')+
  xlab('Level') +
  ylab('Salary')

# Predicting a new resut with linear Regression
y_pred = predict(lin_reg,data.frame(Level = 6.5))

# Predicting a new Result with  Polynomial Regression
y_pred = predict(poly_reg,data.frame(Level = 6.5,Level2 = 6.5^2,Level3 = 6.5^3,Level4 = 6.5^4))
