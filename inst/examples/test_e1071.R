# Examples Practice for Package e1071
# http://www.csie.ntu.edu.tw/~cjlin/libsvm/R_example.html

require(e1071)

################ A Simple Example ################
x <- array(data = c(0,0,1,1,0,1,0,1),dim=c(4,2))
y <- factor(c(1,-1,-1,1))
model <- svm(x,y,type="C-classification")
predict(model,x)

################ A Comprehensive Example ################
