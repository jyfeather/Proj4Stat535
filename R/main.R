#-------------------------------------------------------------------- 
#                         Parameters Preset  
#--------------------------------------------------------------------
rm(list = ls())
library(e1071)

#-------------------------------------------------------------------- 
#                         Data Input  
#--------------------------------------------------------------------
dat.orig <- read.delim(file = "./data/train.txt", header = F)
x.orig <- dat.orig[,-78]
y.true <- dat.orig[,78]
table(y.true) # 99144 0, 856 1, imbalance dataset
x.orig <- as.matrix(x.orig)
y.true <- as.matrix(y.true)

#-------------------------------------------------------------------- 
#                         Classification  
# Cross validation
# Bootstrap
# other kernels
# issue: time consuming
#--------------------------------------------------------------------
x.test <- x.orig[1:50000,]
y.test <- y.true[1:50000]

# Feature Selection, group lasso
library(grplasso)
groupindex <- c(rep(1,7), rep(2,7), rep(3,7), rep(4,7), rep(5,7),
                rep(6,7), rep(7,7), rep(8,7), rep(9,7), rep(10,7), rep(11,7))
lambda <- grplasso(x = x.orig, y = y.true, index = groupindex, penscale = sqrt, 
                   model = LogReg(), center = F, lambda = 10)


# Model Selection, time consuming
tuned <- tune.svm(x = x.orig, y = y.true, gamma = 10^(-6:-1), cost = 10^(1:2)) # CV

# Learn Model
model.gaussian <- svm(x.orig, y.true, kernel = "radial", probability = TRUE) # Gaussian Kernel
#model.linear <- svm(x.train, y.train, kernel = "linear", probability = TRUE)
#model.ploynomial <- svm(x.train, y.train, kernel = "polynomial", probability = TRUE)
#model.sigmoid <- svm(x.train, y.train, kernel = "sigmoid", probability = TRUE)

pred <- predict(model.gaussian, x.test, probability = TRUE)
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
