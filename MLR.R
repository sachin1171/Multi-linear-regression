############################# problem 1 ##########################
# Do transformations for getting better predictions of profit and 
# make a table containing R^2 value for each prepared model.

# Loading the data
opt_profit<-read.csv(file.choose())
View(opt_profit)
summary(opt_profit)
str(opt_profit)
# Defining State as a factor
opt_profit$State<-factor(opt_profit$State,levels=c('New York','California','Florida'), labels=c(1,2,3))
str(opt_profit)
install.packages("moments")
library(moments)
install.packages("lattice")
library(lattice)
attach(opt_profit)
# Understanding data of R.D. Spend
skewness(R.D.Spend)
kurtosis(R.D.Spend)
dotplot(R.D.Spend)
hist(R.D.Spend)
qqnorm(R.D.Spend)
qqline(R.D.Spend)
boxplot(R.D.Spend)
# R.D. Spend does not exactly follow the normal distribution.
# Mean is greater than median and skewness is +ve. It indicates positively skewed distributed
# However, there is no outliers as evident from the boxplot.

# Understanding data of Administration
skewness(Administration)
kurtosis(Administration)
dotplot(Administration)
hist(Administration)
qqnorm(Administration)
qqline(Administration)
boxplot(Administration)
# This is a negative skewed distributed with -ve skewness. Mean is lower than median value.
# However,there is no outliers

# Understanding Marketing Spend
skewness(Marketing.Spend)
kurtosis(Marketing.Spend)
dotplot(Marketing.Spend)
hist(Marketing.Spend)
qqnorm(Marketing.Spend)
qqline(Marketing.Spend)
boxplot(Marketing.Spend)
# This is negative skewed distributed with negative skewness.
# Median is more than mean which also indicates that it is -ve skewed distributed.
# There is no outlier exists
# Understanding the output -Profit
skewness(Profit)
kurtosis(Profit)
dotplot(Profit)
hist(Profit)
qqnorm(Marketing.Spend)
qqline(Marketing.Spend)
boxplot(Profit)
# Profit is positive skewed distributed with mean greater than median.
# There is no outlier in it

# Relationship of output Profit with other variables & relation among all input variables
pairs(opt_profit)
cor(opt_profit[,-4])
# It indicates that there is a strong correlation i.e. 0.97 between R.D.Spend and Profit
# There is a moderate correlation i.e. 0.75 between Marketing Spend and Profit
# There is a moderate correlation i.e. 0.72 between R.D.Spend and Marketing Spend
# State and Administration do not have any effect on profit or any other variables

# Building Model of Profit with input variables
library(caTools)
model1<-lm(Profit~.,data=opt_profit)
summary(model1)
plot(model1)
# R^2 is 0.95 which is Excellent
# R.D.Spend is found to be significant and others are found to be insignificant
# Check colinearity among input variables
install.packages("car")
library(car)
install.packages(carData)
library(carData)
car::vif(model1)
# VIF values for all variables are found to be less than 10- No colinearity
library(MASS)
stepAIC(model1)
# AIC values were found to be reducing in absence of State and Administration.
# Therefore, these two variables are not significant
residualPlots(model1)
avPlots(model1)
qqPlot(model1)
# There is no trend found in residual plots
# R.D.Spend & Marketing Spend found to have contributions to prediction of Profit
# First Iteration (Removal of State)
model2<-lm(Profit~R.D.Spend+Administration+Marketing.Spend,data=opt_profit)
summary(model2)
# Second Iteration (Removal of Administration)
model3<-lm(Profit~R.D.Spend+Marketing.Spend,data=opt_profit)
summary(model3)
# Third Iteration (Removal of Marketing Spend)
model4<-lm(Profit~R.D.Spend,data=opt_profit)
summary(model4)
# Since there is a decrease in R^2 value by not considering the Marketing Spend.
# Moreover this is also significant at 90% significance level
# Therefore, let's consider both the variables
model5<-lm(Profit~R.D.Spend+Marketing.Spend, data=opt_profit)
summary(model5)
plot(model5)
pred<-predict(model5,interval="predict")
pred1<-data.frame(pred)
cor(pred1$fit,opt_profit$Profit)
plot(pred1$fit,opt_profit$Profit)
# Correlation between predicted and actual found to be strong i.e. 0.97
# For further improvement in the model, we can check data points influencing the model
influenceIndexPlot(model5)
# We have observed the data point 50 is above limits in Diagnosis plots.
# We can make the model by eliminating the influencing data point 50
model6<-lm(Profit~R.D.Spend+Marketing.Spend, data=opt_profit[-50,])
summary(model6)
# R^2 value has improved to 0.96
# Calculation of RMSE
sqrt(sum(model5$residuals^2)/nrow(opt_profit))

model_R_Squared_values <- list(model=NULL,R_squared=NULL,RMSE=NULL)
model_R_Squared_values[["model"]] <- c("model1","model2","model3","model4","model5","model6")
model_R_Squared_values[["R_squared"]] <- c(0.95,0.95,0.95,0.94,0.95,0.96)
model_R_Squared_values[["RMSE"]]<-c(9439,9232,9161,9416,9161,7192)
Final <- cbind(model_R_Squared_values[["model"]],model_R_Squared_values[["R_squared"]],model_R_Squared_values[["RMSE"]])
View(model_R_Squared_values)
View(Final)
# Final model is as given below :
final_model<-lm(Profit~R.D.Spend+Marketing.Spend, data=opt_profit[-50,])
summary(final_model)
pred<-predict(final_model,interval="predict")
pred1<-data.frame(pred)
pred1
cor(pred1$fit,opt_profit[-50,]$Profit)
plot(pred1$fit,opt_profit[-50,]$Profit)
# Final model gives R^2 value 0.96 and correlation with fitting value as 0.98

######################## problem 2 ###########################
install.packages("readr")
library(readr)
Computer_Data <- read.csv(file.choose())
View(Computer_Data)
com <- Computer_Data[,-1]
View(com)
attach(com)

com$cd <- as.integer(factor(com$cd, levels = c("yes", "no"), labels = c(1, 0)))
com$multi <- as.integer(factor(com$multi, levels = c("yes", "no"), labels = c(1, 0)))
com$premium <- as.integer(factor(com$premium, levels = c("yes", "no"), labels = c(1, 0)))
View(com)
attach(com)

str(com)
summary(com) #1st business decision
install.packages("psych")
library(psych)
describe(com) #2nd business decision
plot(com)
pairs(com)
install.packages("GGally")
library(GGally)
install.packages("ggplot2")
library(ggplot2)
ggplot(data=com)+geom_histogram(aes(x=price,),bin=40)

#Mutiple Linear Regression
model_com1 <- lm(com$price~.,data = com)
summary(model_com1)                                  #R^2=0.7756
rmse1 <- sqrt(mean(model_com1$residuals^2))
rmse1                                                #RMSE=275.1298
pred1 <- predict(model_com1, newdata = com)
cor(pred1, com$price)                                #Accuracy=0.8806631
vif(model_com1)
avPlots(model_com1)
influenceIndexPlot(model_com1, grid = T, id = list(n=10, cex=1.5, col="blue"))
influence.measures(model_com1)
influencePlot(model_com1)
qqPlot(model_com1)

#Removing influencing Observations
model_com2 <- lm(price~., data = com[-c(1441, 1701),])
summary(model_com2)                                  #R^2=0.7777
rmse2 <- sqrt(mean(model_com2$residuals^2))
rmse2                                                #RMSE=272.8675
pred2 <- predict(model_com2, newdata = com)
vif(model_com2)
cor(pred2, com$price)                                #Accuracy=0.8806566
avPlots(model_com2)
qqPlot(model_com2)

#Applying Logarithmic Transformation
x <- log(com[, -1])
log_com <- data.frame(com[,1], x)
colnames(log_com)
attach(log_com)
View(log_com)
model_com3 <- lm(com...1.~speed+hd+ram+screen+cd+multi+premium+ads+trend, data=log_com)
summary(model_com3)                                   #R^2=0.7426
rmse3 <- sqrt(mean(model_com3$residuals^2))
rmse3                                                 #RMSE=294.653
pred3 <- predict(model_com3, newdata = log_com)
cor(pred3, log_com$com...1.)                          #Accuracy=0.8617343
vif(model_com3)
avPlots(model_com3)
qqPlot(model_com3)
influenceIndexPlot(model_com3, grid = T, id = list(n=10, cex=1.5, col="blue"))
influence <- as.integer(rownames(influencePlot(model_com3, grid = T, id = list(n=10, cex=1.5, col="blue"))))
influence

#Log Transformation with Removing Influencial Observations
model_com4 <- lm(com...1.~speed+hd+ram+screen+cd+multi+premium+ads+trend, data = log_com[-c(1441, 1701)])
summary(model_com4)                                   #R^2=0.7426
rmse4 <- sqrt(mean(model_com4$residuals^2))
rmse4                                                 #RMSE=294.653
pred4 <- predict(model_com4, newdata = log_com)
cor(pred4, log_com$com...1.)                          #Accuracy=0.8617343
avPlots(model_com4)
qqPlot(model_com4)

#model_com2 has the best model with high R^2 value and less RMSR
plot(model_com2)

############################ problem 3############################
# Loading the data
Toyota_Corolla<-read.csv(file.choose())
View(Toyota_Corolla)
Toyota_Corolla1<-Toyota_Corolla[,-c(1,2)]
View(Toyota_Corolla1)
str(Toyota_Corolla1)
summary(Toyota_Corolla1)
# It shows that cylinder is constant, which does not give any variance.
# The variable cylinder can be eliminated from the data set for analysis
Toyota_Corolla2<-Toyota_Corolla1[,-6]
View(Toyota_Corolla2)
# Check for correlation between output and input variables among all input variables
pairs(Toyota_Corolla2)
cor(Toyota_Corolla2)
# There is a negative strong correlation found between price and age(-0.876)
# There is a negative moderate correlation found between price and km(-0.57)
# There is a positive correlation found between price and weight(0.58)
# There is a positive correlation between age and km (0.50)
attach(Toyota_Corolla2)
model1<-lm(Price ~. , data=Toyota_Corolla2)
summary(model1)
# R^2 value is observed 0.86 and Door variable was found insignificant
plot(model1)

install.packages("car")
library(car)
car::vif(model1)
# VIF values are found to be less than 10. There is no Collinearity observed.
library(MASS)
stepAIC(model1)
# AIC value decreases by removing the insignificant variable i.e. Door
residualPlots(model1)
avPlots(model1)
qqPlot(model1)
sqrt(sum(model1$residuals^2)/nrow(Toyota_Corolla2))

# QQ Plot looks to be normal.Residual plot of Age is showing a trend
model2<-lm(Price ~ Age_08_04+I(Age_08_04^2)+KM+HP+Gears+Weight, data=Toyota_Corolla2)
summary(model2)
# R^2 value improved to 0.88
# All the variables are found to be significant
plot(model2)
residualPlots(model2)
avPlots(model2)
qqPlot(model2)
sqrt(sum(model2$residuals^2)/nrow(Toyota_Corolla2))

# Trend was found in HP variable in the residual plot
# model was further improved by adding HP^2
model3<-lm(Price ~ Age_08_04+I(Age_08_04^2)+KM+HP+I(HP^2)+Gears+Weight, data=Toyota_Corolla2)
summary(model3)
plot(model3)
residualPlots(model3)
avPlots(model3)
qqPlot(model3)
# No trend is observed in residual plot
# But Data points 222 & 602 are found out of the normal plot.
# These data points can be verified in Diagnosis plots
influenceIndexPlot(model3)
sqrt(sum(model3$residuals^2)/nrow(Toyota_Corolla2))

# 222 & 602 data points also observed in cooks' distance plot
# These two influencing points can be removed from the model
model4<-lm(Price ~ Age_08_04+I(Age_08_04^2)+KM+HP+I(HP^2)+Gears+Weight,
           data=Toyota_Corolla2[-c(222,602),])
summary(model4)
# R^2 value improved to 0.89
plot(model4)
residualPlots(model4)
avPlots(model4)
qqPlot(model4)
influenceIndexPlot(model4)
sqrt(sum(model4$residuals^2)/nrow(Toyota_Corolla2[-c(222,602),]))

# Trend was observed in KM residual plot and
# some influencing data points were identified
model5<-lm(Price ~ Age_08_04+I(Age_08_04^2)+KM+HP+I(HP^2)+Gears+Weight+I(Weight^2),
           data=Toyota_Corolla2[-c(148,192,193,222,602,961,524),])
summary(model5)
plot(model5)
residualPlots(model5)
avPlots(model5)
qqPlot(model5)
influenceIndexPlot(model5)
sqrt(sum(model5$residuals^2)/nrow(Toyota_Corolla2[-c(148,192,193,222,602,961,524),]))
# No residual trend observed.
# It is found to be normal in Q-Q Plot
# R^2 is found to be improved to 0.9 and all the input variables are found to be significant.
model_R_Squared_RMSE_values <- list(model=NULL,R_squared=NULL,RMSE=NULL)
model_R_Squared_RMSE_values[["model"]] <- c("model1","model2","model3","model4","model5")
model_R_Squared_RMSE_values[["R_squared"]] <- c(0.86,0.88,0.87,0.89,0.90)
model_R_Squared_RMSE_values[["RMSE"]]<-c(1342,1240,1258,1195,1112)
final_model <- cbind(model_R_Squared_RMSE_values[["model"]],model_R_Squared_RMSE_values[["R_squared"]],model_R_Squared_RMSE_values[["RMSE"]])
View(model_R_Squared_RMSE_values)
View(final_model)
# Final model is as given below :
final_model<-lm(Price ~ Age_08_04+I(Age_08_04^2)+KM+HP+I(HP^2)+Gears+Weight+I(Weight^2),
                data=Toyota_Corolla2[-c(148,192,193,222,602,961,524),])
pred<-predict(final_model,interval="predict")
pred1<-data.frame(pred)
pred1
Toyota_Corolla3<-Toyota_Corolla2[-c(148,192,193,222,602,961,524),]
cor(pred1$fit,Toyota_Corolla3$Price)
# Since there is a consistent value in R^2 and all the variables are significant, We can take it as final model
# R-Square value is 0.90 and Correlation between fitting value with price is 0.95
######################## problem 4#####################
#load the dataset
av_ds <- read.csv(file.choose())
av_ds$X <- NULL #This auto created column, we don't required this
head(av_ds)

library(anytime)
#Convert Date to date
av_ds$Date <- anydate(av_ds$Date)
str(av_ds)
#Dependant Variable : AveragePrice, since this is continous variable so start with MLR
#Step 1: Model Validation: HOLD OUT, divide the data into train and test data, and create model on train_data
library(caret)
#Loading required package: lattice
## Loading required package: ggplot2
index <- createDataPartition(av_ds$AveragePrice, p=0.8, list = F)
train_data <- av_ds[index,]
test_data  <- av_ds[-index,]

av_model_train <- lm(AveragePrice~Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge.Bags+type+year+region, data = train_data)
summary(av_model_train)

#removing the insignificant variable one by one and re-run the model again
av_model_train <- lm(AveragePrice~tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge.Bags+type+year+region, data = train_data)
summary(av_model_train)

#removing the insignificant variable one by one and re-run the model again
av_model_train <- lm(AveragePrice~tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge.Bags+type+year+region, data = train_data)
summary(av_model_train)

#removing the insignificant variable one by one and re-run the model again
av_model_train <- lm(AveragePrice~tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge.Bags+type+year+region, data = train_data)
summary(av_model_train)

#removing the insignificant variable one by one and re-run the model again
av_model_train <- lm(AveragePrice~Total_Bags+Small_Bags+Large_Bags+XLarge.Bags+type+year+region, data = train_data)
summary(av_model_train)

#removing the insignificant variable one by one and re-run the model again
av_model_train <- lm(AveragePrice~Small_Bags+Large_Bags+XLarge.Bags+type+year+region, data = train_data)
summary(av_model_train)

#some region factor levels are not significant but then also we keep this factor, because it also contains the significant levels
summary(av_model_train)

#Now all the variables are significant
#Step 2 : Check for MultiColinearity
library(car)
## Loading required package: carData
vif(av_model_train)

#From the Output above, Date and year are insignificant variables, first remove the variable with highest vif value that is year and re-run the model
av_model_train <- lm(AveragePrice~Large_Bags+XLarge.Bags+type+region, data = train_data)
summary(av_model_train)

#Re-Run Step 2 : Check for MultiColinearity
vif(av_model_train)

#Create the fitted and resi variables in train_data
train_data$fitt <- round(fitted(av_model_train),2)
train_data$resi <- round(residuals(av_model_train),2)
head(train_data)

#Step 3 : Checking the normality of error i.e. resi column from train_data
#There are 2 ways of doing this, as below :
#(a)lillieTest from norTest() package
install.packages("nortest")
library(nortest)
lillie.test(train_data$resi) #We have to accept H0: it is normal

#But from o/p this is not normal
#(b)qqplot
qqnorm(train_data$resi)
qqline(train_data$resi, col = "green")


#From graph also this is not normal
#For MLR model error must be normal, lets do some trnsformaation ton achieve this
#Step 4 : Check for Influencing data in case on non- normal error
#4.(a)
influ <- influence.measures(av_model_train)
#influ
#check for cook.d column and if any value > 1 then remove that value and re-run the model
#4.(b)
influencePlot(av_model_train, id.method = "identical", main = "Influence Plot", sub = "Circle size")
## Warning in plot.window(...): "id.method" is not a graphical parameter
## Warning in plot.xy(xy, type, ...): "id.method" is not a graphical parameter
## Warning in axis(side = side, at = at, labels = labels, ...): "id.method" is
## not a graphical parameter

## Warning in axis(side = side, at = at, labels = labels, ...): "id.method" is
## not a graphical parameter
## Warning in box(...): "id.method" is not a graphical parameter
## Warning in title(...): "id.method" is not a graphical parameter
## Warning in plot.xy(xy.coords(x, y), type = type, ...): "id.method" is not a
## graphical parameter

#from plot 5486 index data is influencing
#Remove 5486 index data from the data set and re-run the model
train_data$fitt <- NULL
train_data$resi <- NULL
train_data <- train_data[-(5485),]


av_model_train <- lm(AveragePrice~Large_Bags+XLarge.Bags+type+region, data = train_data)
summary(av_model_train)

train_data$fitt <- round(fitted(av_model_train),2)
train_data$resi <- round(residuals(av_model_train),2)
head(train_data)

#Repeat 4.(b)
influencePlot(av_model_train, id.method = "identical", main = "Influence Plot", sub = "Circle size")

#Step 5 : Check for Heteroscadicity, H0 : error are randomly spread, we have to accept H0, i.e p-value must be > than 0.05
#(a)plot
plot(av_model_train)

#5.(b) ncvTest
ncvTest(av_model_train, ~Large_Bags+XLarge.Bags+type+region)
## Non-constant Variance Score Test 
## Variance formula: ~ Date + Large.Bags + XLarge.Bags + type + region 
## Chisquare = 3401.212    Df = 57     p = 0
# p = 0 it means there is problem of heteroscadicity 
#Since error are not normal and there is issue of Heteroscadicity, we can transform the dependent variable and this may be resolve these issues.
#take log of Y varibale and re-run the model

train_data$fitt <- NULL
train_data$resi <- NULL
train_data$AveragePrice <- log(train_data$AveragePrice)

av_model_train <- lm(AveragePrice~Large_Bags+XLarge.Bags+type+region, data = train_data)
summary(av_model_train)

#Check again, repeat Step 3 again:
lillie.test(train_data$resi)

qqnorm(train_data$resi)
qqline(train_data$resi, col = "green")


#Still error are not normal
ncvTest(av_model_train, ~Large_Bags+XLarge.Bags+type+region)
## Non-constant Variance Score Test 
## Variance formula: ~ Date + Large.Bags + XLarge.Bags + type + region 
## Chisquare = 2346.56    Df = 57     p = 0
#Still there is issue of Heteroscadicity
#Now re-run the model and re-run the model implementation steps done above.
av_model_train <- lm(AveragePrice~Large_Bags+XLarge.Bags+type+region, data = train_data)
summary(av_model_train)

#Checking the Nomality of error again
train_data$fitt <- fitted(av_model_train)
train_data$resi <- residuals(av_model_train)

lillie.test(train_data$resi) #way 1

#Again H0, ois rejected, and errors are not normal again after the transformation
qqnorm(train_data$resi)   #way 2
qqline(train_data$resi, col = "green")#way 2


#Lest Check the stability of Model using RMSE of Train and Test Data
library(ModelMetrics)
test_data$AveragePrice <- log(test_data$AveragePrice)
test_data$fitt <- predict(av_model_train, test_data)
test_data$resi <- test_data$AveragePrice - test_data$fitt

head(test_data)
head(train_data)

RMSE_train <- RMSE(train_data$AveragePrice, train_data$fitt)
RMSE_test <- RMSE(test_data$AveragePrice, test_data$fitt)

check_stability <- paste0(  round((RMSE_test - RMSE_train)*100,2)," %")

RMSE_train
RMSE_test

check_stability 
# Since the Difference between Test and Train RMSE is less than 10%, so that the model is stable, but not linear acceptable model.
# To make the model Good, we require add more VARIABLES or PREDICTORS, so that the Adjusted R square value must be above 65% or .65