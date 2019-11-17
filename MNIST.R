library(reticulate)
np <- import("numpy")

npz1 <- np$load("noised-MNIST.npz")
npz1$files
x <- npz1$f[["x"]]
y <- npz1$f[["y"]]
x_submission <- npz1$f[["x_submission"]]


i <- 15
num1<- matrix(x[i,],nrow = 28,ncol = 28)
image(1:28, 1:28, num1, col=gray((0:255)/255))
num1 <- apply(num1, 1, rev)
num1 <- apply(num1, 1, rev)
num1 <- apply(num1, 2, rev)
image(1:28, 1:28, num1, col=gray((0:255)/255))


#How images look after removing the pixels = 255 (noise)
view_some_images <- function(x) {
  for (n in c(100:120)) {
    num1<- matrix(x[n,],nrow = 28,ncol = 28)
    num1 <- cbind(apply(num1, 1, rev))
    num1 <- apply(num1, 1, rev)
    num1 <- apply(num1, 2, rev)
    image(1:28, 1:28, num1, col=gray((0:255)/255))
    Sys.sleep(1)
    for (i in c(1:28)) {
      for (j in c(1:28)) {
        if (num1[i,j] == 255) {
          num1[i,j] = 0
        }
       
      }
    }
    image(1:28, 1:28, num1, col=gray((0:255)/255))
    Sys.sleep(1)
  }
}

######################################################

#Preprocess all the data to delete noise AND scale



prep <- function(x) {
  x_toproc <- x
  n1 <- length(x_toproc[,1])
  n2 <- length(x_toproc[1,])
  for (i in c(1:n1)) {
    for (j in c(1:n2)) {
      if (x_toproc[i,j] == 255) {
        x_toproc[i,j] = 0
      }
      else {
        x_toproc[i,j] = x_toproc[i,j]/255
      }
    }
  }
  return (x_toproc)
}

x_proc <- prep(x)
view_some_images(x_proc)

########################
# PRINCIPAL COMPONENTS # (not used)
########################

x_proc <- as.data.frame(x_proc)
pca <- princomp(x_proc)

pc_train <- pca$loadings


pca <- prcomp(x_proc, center = TRUE)
ss <- summary(pca)
ss$importance[3,]
x_proc <- pca$x[,c(1:250)]

##################################################################
##################################################################
##################################################################

library(car)
library(caret)
library(MASS)
library(rgl)
library(klaR)
library(kernlab)
library(class)
library(cclust)
library(TunePareto)
library(e1071)
library(nnet) #only one useful

df <- as.data.frame(cbind(x_proc, y)) 
df[,785] <- as.factor(df[,785])

learn <- sample(1:46900, 5000) 
df_sampled <- df[learn,]


train <- sample(1:46900, 30000) 


tr_df <- df[train,]
test_df <- df[-train,]



###################################
# Multinomial logistic regression # (used)
###################################

model.nnet1<- multinom (y ~ ., data = df, maxit=80, MaxNWts = 10000)
summary(model.nnet1)

tr_pred <- predict(model.nnet1, df[,-785])
test_pred <- predict(model.nnet1, test_df[,c(1:784)])

(tr_t <- table(df[,785], tr_pred)) #confusion table
(tr_acc <- sum(diag(tr_t))/sum(tr_t)) #accuracy

(test_t <- table(test_df[,785], test_pred)) 
(test_acc <- sum(diag(test_t))/sum(test_t))

save(model.nnet1, file = "Model_multinom.RData")

################# PREDICT AND CREATE FILE ################

x_proc_submission <- prep(x_submission)

y_submission <- predict(model.nnet1, x_proc_submission)

filesub<-file("submission.txt")
writeLines(as.character(y_submission), filesub)





########
# NNET # (not used)
########



trc2 <- trainControl (method="repeatedcv", number=2, repeats=1)
(sizes <- c(10,15))

model.10x10CVsize <- train (y ~., data = df_sampled, method='nnet', maxit = 100, trace = FALSE,
                            tuneGrid = expand.grid(.size=sizes,.decay=0), trControl=trc2,
                            MaxNWts = 15000)

trc_none <- trainControl (method="none")


model.10x10CVsize <- train (y ~., data = df_sampled, method='nnet', maxit = 90, trace = TRUE,
                            tuneGrid = expand.grid(.size=15,.decay=0.03), trControl=trc_none,
                            MaxNWts = 15000)



