#-------------------------------------------------------------------- 
#                         Parameters Preset  
#--------------------------------------------------------------------
rm(list = ls())
setwd("C:/Users/jyfea_000/Documents/GitHub/Proj4Stat535")
library(e1071)

#-------------------------------------------------------------------- 
#                         Data Input  
#--------------------------------------------------------------------
dat.orig <- read.delim(file = "./data/train.txt", header = F)
x.orig <- dat.orig[,-78]
y.true <- as.factor(dat.orig[,78])
table(y.true) # 99144 0, 856 1, imbalance dataset

#-------------------------------------------------------------------- 
#                         Classification  
# Cross validation
# Bootstrap
# other kernels
# issue: time consuming
#--------------------------------------------------------------------
x.train <- x.orig[1:50000,]
x.test <- x.orig[1:50000,]
y.train <- y.true[1:50000]
y.test <- y.true[1:50000]

model.gaussian <- svm(x.train, y.train, kernel = "radial", probability = TRUE) # Gaussian Kernel
model.linear <- svm(x.train, y.train, kernel = "linear", probability = TRUE)
model.ploynomial <- svm(x.train, y.train, kernel = "polynomial", probability = TRUE)
model.sigmoid <- svm(x.train, y.train, kernel = "sigmoid", probability = TRUE)

pred <- predict(model.linear, x.test, probability = TRUE)
table(pred, y.test)

#-------------------------------------------------------------------- 
#                         Performance Estimation  
# imbalance dataset, use ROC/AUC as performance evaluator
#--------------------------------------------------------------------
library(ROCR)
y.prob <- attr(pred, "probabilities")[,2]
roc.pred <- prediction(y.prob, y.test)
roc.perf <- performance(roc.pred, "tpr", "fpr")
plot(roc.perf, colorize = TRUE)
auc.tmp <- performance(roc.pred, "auc")
auc <- as.numeric(auc.tmp@y.values)
