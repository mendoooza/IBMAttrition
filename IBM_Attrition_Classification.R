#########################################################
#                                                       #
# Author:     Alex Mendoza                              #
# date:       12/17/2023                                #
# Subject:    Project 3                                 #
# Class:      BDAT 640                                  #
# Section:    01W                                       #
# Instructor: Chris Shannon                             #
# File Name:  ExtraCredit2_Mendoza_Alex.R               #
#                                                       #
#########################################################


# The data set is IBM HR Analytics Employee Attrition & Performance
#
# URL: https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset
# 
# 

##############################
#                            #
#        INSTRUCTIONS        #
#                            #
##############################

# Your boss in the HR department wants you to develop a
# predictive model for employee attrition. You are to
# develop the following models, assess them and make
# a formal recommendation to your boss.

# 1.  Load data from the file employee_attrition.csv and view it.
#     All categorical variables are properly set in the data set
#     on loading.
library(e1071)
library(ROSE)
library(MASS)
library(tree)
library(randomForest)
library(ROCR)


emp.att <- read.csv("employee_attrition.csv")

#Note: it didn't look like the categorical variables are already set as factors.
emp.att$Attrition <- as.factor(emp.att$Attrition)
emp.att$BusinessTravel <- as.factor(emp.att$BusinessTravel)
emp.att$Department <- as.factor(emp.att$Department)
emp.att$Education <- as.factor(emp.att$Education)
emp.att$EducationField <- as.factor(emp.att$EducationField)
emp.att$EnvironmentSatisfaction <- as.factor(emp.att$EnvironmentSatisfaction)
emp.att$Gender <- as.factor(emp.att$Gender)
emp.att$JobInvolvement <- as.factor(emp.att$JobInvolvement)
emp.att$JobLevel <- as.factor(emp.att$JobLevel)
emp.att$JobRole <- as.factor(emp.att$JobRole)
emp.att$JobSatisfaction <- as.factor(emp.att$JobSatisfaction)
emp.att$MaritalStatus <- as.factor(emp.att$MaritalStatus)
emp.att$OverTime <- as.factor(emp.att$OverTime)
emp.att$PerformanceRating <- as.factor(emp.att$PerformanceRating)
emp.att$RelationshipSatisfaction <- as.factor(emp.att$RelationshipSatisfaction)
emp.att$StockOptionLevel <- as.factor(emp.att$StockOptionLevel)
emp.att$WorkLifeBalance <- as.factor(emp.att$WorkLifeBalance)


View(emp.att)
summary(emp.att)

# 2.  Split your data into training and test sets, with two-thirds
#     of your data in the training set and one-third in the test set.

set.seed(1)
index <- sample(1:nrow(emp.att),nrow(emp.att)* (2/3))
train <- emp.att[index,]
test <- emp.att[-index,]

train.majority <- train[train$Attrition == "No",]
train.minority <- train[train$Attrition == "Yes",]

# 3.  Use logistic regression to develop a model, with Attrition
#     as the target variable and all the rest of the variables
#     as predictors. You can use the short-cut formula to specify
#     the target and all predictors:
#

set.seed(1)
fit.glm <- glm(Attrition ~ ., data = train, family = binomial)

pred.glm <- predict(fit.glm, newdata = test, type = "response")

# 4.  Create a confusion matrix using the test data and calculate
#     by hand without using ONLY the sensitivty, specificity and
#     accuracy of the model based on the results of the confusion
#     matrix. For example:
#
#                      Reference
#            Prediction   No  Yes
#                   No  1297  122
#                  Yes    28   60
#
#     Then report the sensitivity, specificity and accuracy
#     of the model as follows, replacing the zeros with
#     your hand calculations from the values from the
#     above table:

pred1 <- rep("No",length(pred.glm))
pred1[pred.glm > 0.5] <- "Yes"
pred1 <- factor(pred1, levels=c("No","Yes"))

table(Prediction = pred1, Reference = test$Attrition)
# 
#
#           Reference
# Prediction  No Yes
#        No  388  45
#        Yes  21  36
# sensitivity : 36 / (36+45) = .4444444
# specificity : 388 / (388+21) = .9486553
# accuracy    : (388+36) / (388+45+36+21) = .8653061

TP1 <- 36
TN1 <- 388
FP1 <- 21
FN1 <- 45

Sens1 <- TP1 / (TP1+FN1)
Spec1 <- TN1 / (TN1+FP1)
Acc1 <- (TN1+TP1)/ (TN1+FN1+TP1+FP1)

logistic.base <- c(
  sensitivity=Sens1,
  specificity=Spec1,
  accuracy=Acc1  
)

# 5.  Evaluate the model and make a decision as to whether a
#     synthetic data can improve the model. Explain to your
#     boss why the synthetic data set is needed if it is needed.
#       Answer: Given the output of the baseline model above, I would recommend
#               that synthetic data could improve the overall effectiveness of the
#               model since we are primarily looking to predict the true positive
#               "yes" value, which is represented poorly with a 44.4% sensitivity.
#               I would explain to my boss that adding synthetic data has the 
#               potential to more often predict rare-cases such as attrition.
#               I would also emphasize that the accuracy stat alone may be
#               misleading since a majority of the cases are true positives.

# 6.  Use R to create three synthetic data sets, one using random
#     under sampling, the second using random oversampling, and the
#     third using the ROSE algorithm.

## Random Undersampling
set.seed(1)
train.undersample.maj <- train.majority[sample(1:nrow(train.majority),
                                               nrow(train.minority),
                                               replace = FALSE),]
# Join undersample to minority
train.undersample <- rbind(train.undersample.maj,train.minority)  
  
## Random Oversampling
set.seed(1)
train.oversample.min <- train.minority[sample(1:nrow(train.minority),
                                              nrow(train.majority),
                                              replace = TRUE),]
# Join Oversample to majority
train.oversample <- rbind(train.oversample.min,train.majority)

## ROSE Algorithm
train.rose <- ROSE(Attrition ~ ., data = train,N = nrow(train.undersample),
                                                        seed = 1)
summary(train.rose)
View(train.rose)
head(train.rose)

# 7.  Pick one of the synthetic data sets to train your models
#     and explain to your boss the advantages and disadvantages
#     of the synthetic data set you picked.
#       Answer: I would explain to my boss that I think the oversampled model
#               may provide us improved results, specifically to the sensitivity
#               metric, and since our training dataset is not tens of thousands
#               of rows it may not be as computationally demanding as other use
#               cases. the primary pro's of oversampling include a proportial 
#               distribution of attrition results (yes/no), and we would retain
#               the original information gained from the majority attrition data.
#               The primary cons we would need to be aware of is the potential for
#               overfitting the minority data due to repeat instances, and larger
#               amounts of data compared to undersampling/ROSE methods, however
#               with our training set only being 1648 obs. it should not be an issue.

# 8.  Using the synthetic data set, train a logistic regression
#     model using all the predictors to predict Attrition. Use
#     the stepAIC function to select a best model. Set the attribute
#     "direction" to "backward". Create a confusion matrix and
#     calculate by hand the sensitivity, specificity and accuracy
#     and load the results in the following variable, replacing the
#     zeros with your hand-calculations:
#
#                 Reference
#       Prediction  No Yes
#               No  334  26
#               Yes  75  55
# sensitivity : 55 / (55+26) = .6790123
# specificity : 334 / (334+75) = .8166259
# accuracy    : (334 + 55) / (334 + 55 + 26 + 75) = .7938776

set.seed(1)
fit.oversample <- glm(Attrition ~ ., data = train.oversample, family = binomial)

summary(fit.oversample)

set.seed(1)
best.oversample <- stepAIC(fit.oversample, scope=list(lower=~ 1),
                         direction="backward") 
summary(best.oversample)

best.lbest <- predict(best.oversample, newdata = test, type = "response")

pred2 <- rep("No",length(best.lbest))
pred2[best.lbest > 0.5] <- "Yes"
pred2 <- factor(pred2, levels=c("No","Yes"))

table(Prediction = pred2, Reference = test$Attrition)

TP2 <- 55
TN2 <- 334
FP2 <- 75
FN2 <- 26
  
Sens2 <- TP2 / (TP2+FN2)
Spec2 <- TN2 / (TN2+FP2)
Acc2 <- (TN2+TP2)/ (TN2+FN2+TP2+FP2)

model.logistic <- c(
  sensitivity=Sens2,
  specificity=Spec2,
  accuracy=Acc2  
)

# 9.  Train a classification tree with the synthetic data set using
#     all the variables to predict Attrition. Use cv.tree to select
#     the optimal tree size, and then use prune.tree to prune your
#     tree. Use the pruned tree to predict the outcome using the
#     test set and create a confusion matrix. From the confusion matrix,
#     calculate the sensitivity, specificity, and accuracy, replacing
#     the zeros with your hand-calculations.
#
#
#               Reference
#     Predicted  No Yes
#           No  268  33
#           Yes 141  48
# sensitivity : 59 / (59+22) = .5925926
# specificity : 228 / (228+181) = .6552567
# accuracy    : (268 + 48) / (268+48+141+33) = .6448980

set.seed(1)
fit.tree <- tree(Attrition ~., data = train.oversample)

set.seed(1)
treelvl <- cv.tree(fit.tree)

plot(treelvl$size, treelvl$dev, type="b",
     main="Optimal Size of Classification Tree",
     xlab="Size",ylab="Deviance")

treesize <- 7

best.tree <- prune.tree(fit.tree, best=treesize)
plot(best.tree)
text(best.tree, pretty=0)
title("Best Tree")

set.seed(1)
pred.tree <- predict(best.tree, newdata=test, type="vector")

head(pred.tree)
pred.tree <- pred.tree[,2]

pred3 <- rep(0, length(pred.tree))
pred3[pred.tree > 0.5] <- 1
pred3 <- factor(pred3, levels=c(0,1),
                      labels=c("No","Yes"))

table(Predicted = pred3, Reference = test$Attrition)


TP3 <- 59
TN3 <- 228
FP3 <- 181
FN3 <- 22
  
Sens3 <- TP3 / (TP3+FN3)
Spec3 <- TN3 / (TN3+FP3)
Acc3 <- (TN3+TP3)/ (TN3+FN3+TP3+FP3)

model.classtree <- c(
  sensitivity=Sens3,
  specificity=Spec3,
  accuracy=Acc3  
)

# 10. Train a random forest model with the synthetic data set
#     and use all variables to predict Attrition. Print out
#     the variable importance using importance() and display
#     the importance using varImpPlot(). Use predict()
#     with the test data to create a set of predictions. Create
#     a confusion matrix and enter your hand-calculations for
#     the sensitivity, specificity and accuracy into the
#     below variable:
#
#                 Reference
#       Prediction  No Yes
#               No  400  65
#               Yes   9  16
# sensitivity : 16 / (16+65) = .1975309
# specificity : 400 / (400+9) = .9779951
# accuracy    : (400 + 16) / (400+16+9+65) = .8489796

set.seed(1)
fit.rf <- randomForest(Attrition ~ .,data = train.oversample, importance = T)
importance(fit.rf)
varImpPlot(fit.rf)

pred.rf <- predict(fit.rf, newdata=test, type="prob")
pred.rf <- pred.rf[,2]

# Convert to a factor
pred4 <- rep(0, length(pred.rf))
pred4[pred.rf > 0.5] <- 1
pred4 <- factor(pred4, levels=c(0,1),
                    labels=c("No","Yes"))

table(Prediction = pred4, Reference = test$Attrition)

TP4 <- 14
TN4 <- 401
FP4 <- 8
FN4 <- 67
  
Sens4 <- TP4 / (TP4+FN4)
Spec4 <- TN4 / (TN4+FP4)
Acc4 <- (TN4+TP4)/ (TN4+FN4+TP4+FP4)

model.randomforest <- c(
  sensitivity=Sens4,
  specificity=Spec4,
  accuracy=Acc4  
)

# 11. Combine the variables model.logistic, model.classtree and
#     model.randomforest into a table and print it.

compare <- rbind(model.logistic, model.classtree, model.randomforest)
compare
# 12. Create an ROC curve featuring the curves of all three models.
#     Give the graphic the following title: "ROC Curves of Attrition
#     Models". Give the model a legend, and in the plot report
#     which model has the maximum Area Under Curve (AUC) and also
#     report the value of the maximum AUC in the plot.

all.probs <- cbind(best.lbest,
                   classtree=pred.tree,
                   rforest=pred.rf)
head(all.probs)

line.colors <- c("brown1","cyan4","darkorchid2")
area.under.curve <- rep(0,ncol(all.probs))

for (i in 1:ncol(all.probs)) {
  rocr.all <- ROCR::prediction(all.probs[,i], test$Attrition)
  perf.all <- ROCR::performance(rocr.all, "sens","spec")
  
  plot(perf.all, col=line.colors[i],
       lwd=2, lty=i,
       add=(i > 1)
  )
  
  auc  <- ROCR::performance(rocr.all, "auc")
  area.under.curve[i] <- auc@y.values[[1]]
} 
title("ROC Curves of Attrition Models")
legend(0.01, 0.7, legend=colnames(all.probs),
       col=line.colors, bty="y",
       lty=1:4, lwd=2, cex=1.3)

best.auc.idx <- which(area.under.curve %in% max(area.under.curve))


#?paste
text(0.14,0.3,
     paste("Best Area Under Curve\n",
           colnames(all.probs)[best.auc.idx],
           "=",
           round(area.under.curve[best.auc.idx]*100,2),
           "%"
     )
)


# 13. Using the table in problem 11 and the ROC curve graphic in
#     problem 12, recommend to your boss which model the department
#     should use and why.
#       Answer: Based on my analysis of the IBM Attrition Data, and specifically
#               the use of a synthetic data by oversampling, I would recommend the
#               use of a logistic regression model over the classification tree,
#               and random forest models. As displayed in the ROC curve chart,
#               the best logistic regression model has the largest area under the
#               curve at 81.63%. The area under the curve represents the trade-off
#               between Specificity and Sensitivity (True Negative/True Positive)
#               predictions. While predicting attrition is considered a rare-event
#               the logistic regression model has a 67.9% sensitivity, 81.6%
#               specificity, and 79.3% accuracy. Accuracy is reportedly lower than
#               the other models, but it can also be a very misleading metric,
#               specifically regarding rare-events.
#               
#               
compare

## END
