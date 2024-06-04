library(dplyr)
library(stringr)
library(RMySQL)
library(dplyr)
library(tidyr)
library(plyr)
library(lubridate)
library(randomForest)
library(caret)
library(nnet)
library(klaR)  # untuk Naive Bayes
library(rpart) # untuk Decision Trees

train <- read.csv('ML/youtube/train.csv', na.strings = c("", "NA", " ", "N/A", "NULL"))
test <- read.csv('ML/youtube/test.csv', na.strings = c("", "NA", " ", "N/A", "NULL"))

train <- dplyr::select(train, -Cabin)
test <- subset(test, select = -Cabin)

print(train[is.na(train$Embarked), ])
train$Embarked[is.na(train$Embarked)] <- names(which.max(table(train$Embarked)))
print(train[rownames(train) == 62,])

print(test[is.na(test$Fare), ])
test$Fare[is.na(test$Fare)] <- mean(test$Fare, na.rm=TRUE)
print(test[153, ])

train$Age[is.na(train$Age)] <- mean(train$Age, na.rm=TRUE)
test$Age[is.na(test$Age)] <- mean(test$Age, na.rm=TRUE)


print(head(train))
print(head(test))
print(sapply(train, class))
print(dim(train))
print(colSums(is.na(train)))
cat('\n\n')
print(sapply(test, class))
print(dim(test))
print(colSums(is.na(test)))
cat('\n\n\n\n')

train$Sex <- ifelse(train$Sex == 'male', 0, 1)
# train$Sex <- case_when(
#     train$Sex == 'male' ~ 0,
#     train$Sex == 'female' ~ 1,
# )
# test$Sex <- dplyr::recode(
#     test$Sex,
#     male = 0,
#     female = 1
# )
test$Sex <- mapvalues(test$Sex, from = c('male', 'female'), to = c('male' = 0, 'female' = 1))


train_dummies <- model.matrix(~ Embarked -1, data = train)
colnames(train_dummies) <- str_replace_all(colnames(train_dummies), 'Embarked', '')
train <- cbind(train, train_dummies)
train <- dplyr::select(train, -Embarked)

test_dummies <- model.matrix(~ Embarked -1, data = test)
colnames(test_dummies) <- str_replace_all(colnames(test_dummies), 'Embarked', '')
test <- cbind(test, test_dummies)
test <- subset(test, select = -Embarked)

# print(head(train_dummies))
# print(head(test_dummies))

train$title <- str_extract(train$Name, "\\b[A-Za-z]+(?=\\.)")
test$title <- str_extract(test$Name, "\\b[A-Za-z]+(?=\\.)")

title_mapping <- c(
    "Mr" = 0, "Miss" = 1, "Mrs" = 2,
    "Master" = 3, "Dr" = 3, "Rev" = 3, "Col" = 3, "Major" = 3, "Mlle" = 3, "Countess" = 3,
    "Ms" = 3, "Lady" = 3, "Jonkheer" = 3, "Don" = 3, "Dona" = 3, "Mme" = 3, "Capt" = 3, "Sir" = 3
)
train$title <- recode(train$title, !!!title_mapping, .default = NULL)
test$title <- mapvalues(test$title, from= names(title_mapping), to = title_mapping)

train <- dplyr::select(train, -PassengerId , -Name, -Ticket)
test <- subset(test, select = -c(PassengerId, Name, Ticket))

print(head(train))
print(head(test))

train_data <- subset(train, select = -Survived)
target <- factor(train$Survived)

ctrl <- trainControl(method="cv", number=5)
model <- train(train_data, target, method="rf", trControl=ctrl)
print('accuracy mean')
print(model$result)
print(mean(model$result$Accuracy))

result <- predict(model, test)
print(result)

df_test <- read.csv("ML/youtube/test.csv", na.strings = c("", "NA", " ", "N/A", "NULL"))
df <- data.frame(
    Name = df_test$Name,
    Survived = result
)
print(head(df))