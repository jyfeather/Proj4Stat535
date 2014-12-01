#-------------------------------------------------------------------- 
#                         Parameters Preset  
#--------------------------------------------------------------------
rm(list = ls())
library(e1071)

#-------------------------------------------------------------------- 
#                         Defined Functions  
# 1. FeatureExpansion(dat): generate new features
# 2. TrainReduction(train, n1, n2): reduce training set sample size
#--------------------------------------------------------------------
FeatureExpansion <- function(dat) {
  # origin: 11 features * 7 periods
  # plus min, mean, median, max, daily increasment
  n <- nrow(dat); p <- ncol(dat); t <- 7
  for (i in 1:(p/7)) {
    tmp <- dat[,((i-1)*7+1):(7*i)]
    #dat <- cbind(dat, apply(tmp, 1, function(x)return(min(x))))
    #dat <- cbind(dat, apply(tmp, 1, function(x)return(max(x))))
    dat <- cbind(dat, apply(tmp, 1, function(x)return(mean(x))))
    #dat <- cbind(dat, apply(tmp, 1, function(x)return(median(x))))
  }  
  # appended after 77 + 4*11 = 121
  for (i in 1:(p/7)) {
    tmp <- dat[,((i-1)*7+1):(7*i)]
    for (j in 2:7) {
      dat <- cbind(dat, tmp[,j] - tmp[,j-1])
    }
  }
  return(dat)
}

TrainReduction <- function(train, y) {
  pos <- which(y == 1)  # keep the imbalanced dataset
  train.new <- train[-pos,]; y.new <- y[-pos]
  train.new.clusters <- kmeans(train.new, centers = 10)
  clu.del <- c()
  for (i in 1:10) { # delete clusters < 500
    size <- train.new.clusters$size[i]
    if (size <= 500) {
      clu.del <- c(clu.del, which(train.new.clusters$cluster == i))
    } else {
      clus <- which(train.new.clusters$cluster == i)
      clu.del <- c(clu.del, sample(clus, size * 0.7))
    }
  }
  train.new <- train.new[-clu.del,]; y.new <- y.new[-clu.del]
  train.new <- rbind(train.new, train[pos,]); y.new <- c(y.new, y[pos])
  dat <- cbind(train.new, y.new)
  return(dat)
}
#-------------------------------------------------------------------- 
#                         Data Input  
#--------------------------------------------------------------------
dat.orig <- read.delim(file = "./data/train.txt", header = F)
dat.test <- read.delim(file = "./data/test.txt", header = F)
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
# Feature Generation, 187 features in total
x.train <- FeatureExpansion(x.orig)

x.test <- x.train[1:50000,]
y.test <- y.true[1:50000]

# Train dataset dimension reduction
dat.red <- TrainReduction(x.train, y.true)
x.train <- dat.red[,1:154]; y.true <- dat.red[,155]

# Model Selection, time consuming
#tuned <- tune.svm(x = x.train[1:10000,], y = y.true[1:10000,], 
#                  gamma = 10^(-2:-1), cost = 10^(1:2),
#                  tunecontrol = tune.control(sampling = "cross", cross = 5)) # CV

# Learn Model
model.gaussian <- svm(x.train[,1:77], y.true, kernel = "radial", probability = TRUE) # Gaussian Kernel
#model.linear <- svm(x.train[,1:77], y.true, kernel = "linear", probability = TRUE)
#model.ploynomial <- svm(x.train[,1:77], y.true, kernel = "polynomial", probability = TRUE)
#model.sigmoid <- svm(x.train, y.train, kernel = "sigmoid", probability = TRUE)

pred.gaussian <- predict(model.gaussian, x.test[,1:77], decision.values = TRUE, probability = TRUE)
table(pred.gaussian, y.test)

#-------------------------------------------------------------------- 
#                         Performance Estimation  
# imbalance dataset, use ROC/AUC as performance evaluator
#--------------------------------------------------------------------
pred <- pred.gaussian
library(ROCR)
roc.pred <- prediction(attributes(pred)$decision.values, y.test)
roc.perf <- performance(roc.pred, "tpr", "fpr")
plot(roc.perf, colorize = TRUE)
auc.tmp <- performance(roc.pred, "auc")
auc <- as.numeric(auc.tmp@y.values)

#-------------------------------------------------------------------- 
#                         Testing  
#--------------------------------------------------------------------
dat.test <- FeatureExpansion(dat.test)
res.gaussian <- predict(model.gaussian, dat.test[,1:77], decision.values = TRUE, probability = TRUE)
output <- attributes(res.gaussian)$decision.values
write.csv(output[1:10000], file = "./result/f2.out", row.names = FALSE, col.names = FALSE)
