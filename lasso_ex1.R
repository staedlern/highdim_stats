#' ---
#' title: "Add your title"
#' subtitle: "Add your subtitle"
#' author: "Nicolas Staedler"
#' date: "`r Sys.Date()`"
#' output:
#'  html_document:
#'     mathjax: default
#'     number_sections: yes
#'     toc: yes
#'     toc_depth: 3
#'     toc_float:
#'       collapsed: true
#'       smooth_scroll: true
#'     code_folding: hide
#'  pdf_document:
#'    number_sections: yes
#'    toc: yes
#'    toc_depth: 3
#' ---
#'

#+ results = 'hide', message = FALSE
# global chunk options
library(knitr)
opts_chunk$set(echo = TRUE, eval = TRUE, warning = FALSE, message = FALSE, fig.align = 'center', cache = FALSE,
               fig.width=5,fig.height = 5)

# load packages
library(tidyverse)
library(glmnet)

#' ***
#' # Content
#'
#' Add some background info...
#'



#' ***
#' # Results
#'
#' Data simulation of training and test data.
#+
set.seed(1)
n <- 50
p <- 1000
beta <- c(1,rep(0,p-1))
xtrain <- matrix(rnorm(n*p),n,p)
ytrain <- xtrain%*%beta+rnorm(n,sd=1)
xtest <- matrix(rnorm(n*p),n,p)
ytest <- xtest%*%beta+rnorm(n,sd=1)

#' 
#' Linear model with one predictor.
#+
fit.lm <- lm(ytrain~xtrain[,1])
yhat.lm <- predict(fit.lm)
ypred.lm <- cbind(rep(1,n),xtest[,1])%*%coef(fit.lm)

#'
#' Linear model with n-1 predictors.
#+
fit.lm2 <- lm(ytrain~xtrain[,1:(40)])
yhat.lm2 <- predict(fit.lm2)
ypred.lm2 <- cbind(rep(1,n),xtest[,1:(40)])%*%coef(fit.lm2)

#'
#' Lasso regression.
#+
fit.lasso <- glmnet(x,y)
cvfit.lasso <- cv.glmnet(x, y)

#' Trace plot.
#+
plot(fit.lasso,xvar="lambda",cex=3,lwd=2)

#' Cross-validation plot.
#+
plot(cvfit.lasso)


#'
#+ 
dtrain <- cbind(ytrain,yhat.lm,yhat.lm2,data.frame(xtrain))
dtest <- cbind(ytest,ypred.lm,ypred.lm2,data.frame(xtest))
dtrain$yhat.lasso <- predict(cvfit.lasso, newx = xtrain, s = "lambda.min")
dtest$ypred.lasso <- predict(cvfit.lasso, newx = xtest, s = "lambda.min")


#' Fitting.
#+ 
dtrain%>%
  arrange(X1)%>%
  ggplot(.,aes(x=X1,y=ytrain))+
  geom_line(aes(x=X1,y=yhat.lm),col="blue",lwd=1)+
  #geom_line(aes(x=X1,y=yhat.lm2),col="red",lty=2)+
  #geom_line(aes(x=X1,y=yhat.lasso),col="green",lwd=1)+
  geom_point(size=2)+
  theme_bw()+
  theme(text=element_text(size=20))+
  ylab("Y")
dtrain%>%
  arrange(X1)%>%
  ggplot(.,aes(x=X1,y=ytrain))+
  geom_line(aes(x=X1,y=yhat.lm),col="blue",lwd=1)+
  geom_line(aes(x=X1,y=yhat.lm2),col="red",lty=2)+
  geom_line(aes(x=X1,y=yhat.lasso),col="green",lwd=1)+
  geom_point(size=2)+
  theme_bw()+
  theme(text=element_text(size=20))+
  ylab("Y")

#' Prediction
#+
dtest%>%
  arrange(X1)%>%
  ggplot(.,aes(x=X1,y=ytest))+
  geom_line(aes(x=X1,y=ypred.lm),col="blue",lwd=1)+
  geom_point(size=2)+
  theme_bw()+
  theme(text=element_text(size=20))+
  ylab("Y")

dtest%>%
  arrange(X1)%>%
  ggplot(.,aes(x=X1,y=ytest))+
  geom_line(aes(x=X1,y=ypred.lm),col="blue",lwd=1)+
  geom_line(aes(x=X1,y=ypred.lm2),col="red",lty=2)+
  geom_line(aes(x=X1,y=ypred.lasso),col="green",lwd=1)+
  geom_point(size=2)+
  theme_bw()+
  theme(text=element_text(size=20))+
  ylab("Y")


#' ***
#' **SESSION INFO**
#+ collapse = TRUE
sessionInfo()

