library(dplyr)
library(stringr)
library(randomForest)
library(caret)
library(nnet)
library(klaR)  # untuk Naive Bayes
library(rpart) # untuk Decision Trees



train <- read.csv("Praktek/Youtube/train.csv", na.strings = c("", "NA", " ", "N/A", "NULL"))
test <- read.csv("Praktek/Youtube/test.csv", na.strings = c("", "NA", " ", "N/A", "NULL"))

train <- subset(train, select = -Cabin)
test <- subset(test, select = -Cabin)

print(train[is.na(train$Embarked), ])
print(table(train$Embarked))
modus <- names(which.max(table(train$Embarked)))
train$Embarked[is.na(train$Embarked)] <- modus
print(train[rownames(train) == 62, ])

print(test[is.na(test$Fare), ])
mean_fare <- mean(test$Fare, na.rm = TRUE)
test$Fare[is.na(test$Fare)] <- mean_fare
print(test[row.names(test) == 153, ])

print(head(train[is.na(train$Age), ]))
mean_age1 <- mean(train$Age, na.rm=TRUE)
train$Age[is.na(train$Age)] <- mean_age1
print(train[rownames(train) == 6, ])


print(head(test[is.na(test$Age), ]))
mean_age2 <- mean(test$Age, na.rm = TRUE)
test$Age[is.na(test$Age)] <- mean_age2
print(test[rownames(test) == 11, ])
print(test[11, ])



print(dim(train))
print(dim(test))
print(colSums(is.na(train)))
print(colSums(is.na(test)))
print(sapply(train, class))
print(sapply(test, class))
print(sapply(train, function(x) length(unique(na.omit(x)))))
print(sapply(test, function(x) length(unique(na.omit(x)))))
result_train <- train %>% summarise(across(everything(), ~ n_distinct(.[!is.na(.)])))
result_test <- test %>% summarise(across(everything(), ~ n_distinct(.[!is.na(.)])))
print(result_train)
print(result_test)

train$Sex <- ifelse(train$Sex == 'male', 1, ifelse(train$Sex == 'female', 0, NA))
test$Sex <- ifelse(test$Sex == 'male', 1, ifelse(test$Sex == 'female', 0, NA))

data_train_embarked <- model.matrix(~ Embarked -1, data = train)
data_test_embarked <- model.matrix(~ Embarked -1, data = test)

data_train_embarked <- as.data.frame(data_train_embarked)
data_test_embarked <- as.data.frame(data_test_embarked)

colnames(data_train_embarked) <- gsub("Embarked", "", colnames(data_train_embarked))
colnames(data_test_embarked) <- gsub("Embarked", "", colnames(data_test_embarked))

train <- cbind(train, data_train_embarked)
train <- subset(train, select = -Embarked)
test <- cbind(test, data_test_embarked)
test <- subset(test, select=-Embarked)

train$title <- str_extract(train$Name, "\\b[A-Za-z]+(?=\\.)")
test$title <- str_extract(test$Name, "\\b[A-Za-z]+(?=\\.)")

title_mapping <- c(
    "Mr" = 0, "Miss" = 1, "Mrs" = 2,
    "Master" = 3, "Dr" = 3, "Rev" = 3, "Col" = 3, "Major" = 3, "Mlle" = 3, "Countess" = 3,
    "Ms" = 3, "Lady" = 3, "Jonkheer" = 3, "Don" = 3, "Dona" = 3, "Mme" = 3, "Capt" = 3, "Sir" = 3
)
train <- train %>% mutate(title = recode(title, !!!title_mapping, .default=NULL))
test <- test %>% mutate(title = recode(title, !!!title_mapping, .default=NULL))

train <- subset(train, select = -c(Name, Ticket, PassengerId ))
test <- subset(test, select = -c(Name, Ticket, PassengerId ))

print(head(train))
print(head(test))


train_data <- subset(train, select= -Survived)
target <- factor(train$Survived)


# Random Forest (method = "rf")
# Neural Network (method = "nnet")
# Decision Trees (method = "rpart")
# Naive Bayes (method = "nb")

ctrl <- trainControl(method="cv", number=5)
model <- train(train_data, target, method="rf", trControl=ctrl)
# model <- train(train_data, target, method="nnet", trControl=ctrl)
# model <- train(train_data, target, method="rpart", trControl=ctrl)
# model <- train(train_data, target, method="nb", trControl=ctrl)
print('accuracy mean')
print(model$result)
print(mean(model$result$Accuracy))

pred <- predict(model, test)
print(head(pred))

df_test <- read.csv("Praktek/Youtube/test.csv", na.strings = c("", "NA", " ", "N/A", "NULL"))
hasil <- data.frame(
    PassengerId = df_test$PassengerId,
    Survived = pred
)
print(head(hasil))

write.csv(hasil, "Praktek/Youtube/hasil.csv", row.names = FALSE)

print(dim(train))
print(dim(test))
print(colSums(is.na(train)))
print(colSums(is.na(test)))
print(sapply(train, class))
print(sapply(test, class))
print(sapply(train, function(x) length(unique(na.omit(x)))))
print(sapply(test, function(x) length(unique(na.omit(x)))))
result_train <- train %>% summarise(across(everything(), ~ n_distinct(.[!is.na(.)])))
result_test <- test %>% summarise(across(everything(), ~ n_distinct(.[!is.na(.)])))
print(result_train)
print(result_test)

train$Sex <- ifelse(train$Sex == 'male', 1, ifelse(train$Sex == 'female', 0, NA))
test$Sex <- ifelse(test$Sex == 'male', 1, ifelse(test$Sex == 'female', 0, NA))

data_train_embarked <- model.matrix(~ Embarked -1, data = train)
data_test_embarked <- model.matrix(~ Embarked -1, data = test)

data_train_embarked <- as.data.frame(data_train_embarked)
data_test_embarked <- as.data.frame(data_test_embarked)

colnames(data_train_embarked) <- gsub("Embarked", "", colnames(data_train_embarked))
colnames(data_test_embarked) <- gsub("Embarked", "", colnames(data_test_embarked))

train <- cbind(train, data_train_embarked)
train <- subset(train, select = -Embarked)
test <- cbind(test, data_test_embarked)
test <- subset(test, select=-Embarked)

train$title <- str_extract(train$Name, "\\b[A-Za-z]+(?=\\.)")
test$title <- str_extract(test$Name, "\\b[A-Za-z]+(?=\\.)")

title_mapping <- c(
    "Mr" = 0, "Miss" = 1, "Mrs" = 2,
    "Master" = 3, "Dr" = 3, "Rev" = 3, "Col" = 3, "Major" = 3, "Mlle" = 3, "Countess" = 3,
    "Ms" = 3, "Lady" = 3, "Jonkheer" = 3, "Don" = 3, "Dona" = 3, "Mme" = 3, "Capt" = 3, "Sir" = 3
)
train <- train %>% mutate(title = recode(title, !!!title_mapping, .default=NULL))
test <- test %>% mutate(title = recode(title, !!!title_mapping, .default=NULL))

train <- subset(train, select = -c(Name, Ticket, PassengerId ))
test <- subset(test, select = -c(Name, Ticket, PassengerId ))

print(head(train))
print(head(test))


train_data <- subset(train, select= -Survived)
target <- factor(train$Survived)


# Random Forest (method = "rf")
# Neural Network (method = "nnet")
# Decision Trees (method = "rpart")
# Naive Bayes (method = "nb")

ctrl <- trainControl(method="cv", number=5)
model <- train(train_data, target, method="rf", trControl=ctrl)
# model <- train(train_data, target, method="nnet", trControl=ctrl)
# model <- train(train_data, target, method="rpart", trControl=ctrl)
# model <- train(train_data, target, method="nb", trControl=ctrl)
print('accuracy mean')
print(model$result)
print(mean(model$result$Accuracy))

pred <- predict(model, test)
print(head(pred))

df_test <- read.csv("Praktek/Youtube/test.csv", na.strings = c("", "NA", " ", "N/A", "NULL"))
hasil <- data.frame(
    PassengerId = df_test$PassengerId,
    Survived = pred
)
print(head(hasil))

write.csv(hasil, "Praktek/Youtube/hasil.csv", row.names = FALSE)

