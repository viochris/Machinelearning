library(dplyr)
library(ggplot2)
library(stringr)
library(gridExtra)
library(RMySQL)
library(dplyr)
library(tidyr)
library(plyr)
library(glmnet)
library(lubridate)
library(tidyverse)
library(randomForest)
library(caret)
library(patchwork)
library(nnet)
library(klaR) # untuk Naive Bayes
library(rpart) # untuk Decision Trees

train <- read.csv("ML/youtube/lat6-house/train.csv")
test <- read.csv("ML/youtube/lat6-house/test.csv")
print(head(train))
print(head(test))
print(dim(train))
print(dim(test))
print(colSums(is.na(train)))
print(colSums(is.na(test)))
cat("\n\n")

train_posted <- model.matrix(~ POSTED_BY -1, data = train)
test_posted <- model.matrix(~ POSTED_BY -1, data = test)

colnames(train_posted) <- str_replace_all(colnames(train_posted), 'POSTED_BY', '')
colnames(test_posted) <- gsub('POSTED_BY', '',colnames(test_posted))

print(head(train_posted))
print(head(test_posted))

train <- cbind(train, train_posted)
train <- subset(train, select = -POSTED_BY)
test <- cbind(test, test_posted)
test <- subset(test, select = -POSTED_BY)


train$BHK_OR_RK <- ifelse(train$BHK_OR_RK == 'BHK', 0, 1)
test$BHK_OR_RK <- ifelse(test$BHK_OR_RK == 'BHK', 0, 1)

train$alamat <- sapply(strsplit(train$ADDRESS, ','), function(x) tail(x, 1))
train <- dplyr::select(train, -ADDRESS)
test$alamat <- str_split(test$ADDRESS, ',') %>% map_chr(last)
test <- subset(test, select = -ADDRESS)

train$alamat <- case_when(
    train$alamat %in% c('Bangalore', 'Mysore', 'Chennai', 'Mumbai', 'Delhi', 'Kolkata') ~ 0,
    train$alamat %in% c('Pune', 'Hyderabad', 'Ahmedabad', 'Surat', 'Jaipur') ~ 1,
    .default = 2
)

test$alamat <- case_when(
    test$alamat %in% c('Bangalore', 'Mysore', 'Chennai', 'Mumbai', 'Delhi', 'Kolkata') ~ 0,
    test$alamat %in% c('Pune', 'Hyderabad', 'Ahmedabad', 'Surat', 'Jaipur') ~ 1,
    .default = 2
)

print(head(train))
print(head(test))

train_data <- dplyr::select(train, -TARGET.PRICE_IN_LACS.)
target <- train$TARGET.PRICE_IN_LACS.

print(head(train_data))
print(head(target))


# lm_model <- lm(target ~ ., data = train_data)
# predicted_values <- predict(lm_model, newdata = test)
# print(head(predicted_values))


set.seed(42)
ctrl <- trainControl(method="cv", number=5)
# lm_model <- train(x = x_train, y = y_train, method="lm", trControl=ctrl, metric = "Rsquared")
# lm_model <- train(x = x_train, y = y_train, method="lm", trControl=ctrl)
lm_model <- train(x = train_data, y = target, method="lm", trControl=ctrl, metric = "Rsquared", seeds = list(seed = sample.int(1000, 5)))
print('accuracy mean')
print(lm_model$result)
print(lm_model$results$Rsquared)
print(mean(lm_model$results$Rsquared))
cat('\n\n')


glm_model <- train(x = train_data, y = target, method="glm", trControl=ctrl, metric = "Rsquared")
print('accuracy mean')
print(glm_model$result)
print(glm_model$results$Rsquared)
print(mean(glm_model$results$Rsquared))
cat('\n\n')


glmnet_model <- train(x = train_data, y = target, method="glmnet", trControl=ctrl, metric = "Rsquared")
print('accuracy mean')
print(glmnet_model$result)
print(glmnet_model$results$Rsquared)
print(mean(glmnet_model$results$Rsquared))
cat('\n\n')

# glmnet_model <- train(x = train_data, y = target, method="rf", trControl=ctrl, metric = "Rsquared")
# print('accuracy mean')
# print(glmnet_model$result)
# print(glmnet_model$results$Rsquared)
# print(mean(glmnet_model$results$Rsquared))
# cat('\n\n')


predicted_values <- predict(lm_model, newdata = test)
print(head(predicted_values))


