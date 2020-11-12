library(quantmod)  
library(nnet)
library(caret)
## Loading required package: lattice
## Loading required package: ggplot2
t <- seq(0,20,length=200)                       # time stamps
y <- 1 + 3*cos(4*t+2) +.2*t^2 + rnorm(200)      # the time series we want to predict
dat <- data.frame( y, x1=Lag(y,1), x2=Lag(y,2)) # create a triple with lagged values
names(dat) <- c('y','x1','x2')
head(dat)
##               y            x1            x2
## 1 -1.0628369855            NA            NA
## 2 -0.9461638315 -1.0628369855            NA
## 3 -2.4148119350 -0.9461638315 -1.0628369855
## 4  0.1578938481 -2.4148119350 -0.9461638315
## 5 -0.4744434653  0.1578938481 -2.4148119350
## 6 -0.1407492830 -0.4744434653  0.1578938481
#Fit model
model <- train(y ~ x1+x2, dat, method='nnet', linout=TRUE, trace = FALSE)
## Warning in nominalTrainWorkflow(x = x, y = y, wts = weights, info =
## trainInfo, : There were missing values in resampled performance measures.
ps <- predict(model, dat)

#Examine results

plot(t,y,type="l",col = 2)
lines(t[-c(1:2)],ps, col=3)
legend(1.5, 80, c("y", "pred"), cex=1.5, fill=2:3)


#Caret can make train/test sets. Using Caret for the isis dataset:
  
inTrain <- createDataPartition(y=iris$Species, p=0.75, list=FALSE)   # We wish 75% for the trainset 

train.set <- iris[inTrain,]
test.set  <- iris[-inTrain,]
nrow(train.set)/nrow(test.set) # should be around 3
## [1] 3.166666667
model <- train(Species ~ ., train.set, method='nnet', trace = FALSE) # train
# we also add parameter 'preProc = c("center", "scale"))' at train() for centering and scaling the data
prediction <- predict(model, test.set[-5])                           # predict
table(prediction, test.set$Species)                                  # compare
##             
## prediction   setosa versicolor virginica
##   setosa         12          0         0
##   versicolor      0         12         1
##   virginica       0          0        11
# predict can also return the probability for each class:
prediction <- predict(model, test.set[-5], type="prob")  
head(prediction)
##          setosa     versicolor      virginica
## 3  0.9833909273 0.012318908806 0.004290163909
## 13 0.9804309544 0.014774233586 0.004794812024
## 17 0.9876425117 0.008858596249 0.003498892061
## 30 0.9766536305 0.017949582649 0.005396786813
## 31 0.9751143511 0.019253775537 0.005631873333
## 34 0.9884559227 0.008206873188 0.003337204088