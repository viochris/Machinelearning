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

train <- read.csv('ML/youtube/lat1-titanic/train.csv', na.strings = c("", "NA", " ", "N/A", "NULL"))
test <- read.csv('ML/youtube/lat1-titanic/test.csv', na.strings = c("", "NA", " ", "N/A", "NULL"))

train <- dplyr::select(train, -Cabin)
test <- subset(test, select = -Cabin)

print(train[is.na(train$Embarked), ])
train$Embarked[is.na(train$Embarked)] <- names(which.max(table(train$Embarked)))
print(train[rownames(train) == 62, ])

test$Fare[is.na(test$Fare)] <- mean(test$Fare, na.rm = TRUE)
train$Age[is.na(train$Age)] <- mean(train$Age, na.rm = TRUE)
test$Age[is.na(test$Age)] <- mean(test$Age, na.rm = TRUE)

print(head(train))
print(head(test))
cat(dim(train), dim(test), '\n\n')
print(colSums(is.na(train)))
print(colSums(is.na(test)))

train$Sex <- case_when(
    train$Sex == 'male' ~ 0, 
    train$Sex == 'female' ~ 1
)
# test$Sex <- ifelse(test$Sex == 'male', 0, 1)
# test <- test %>% mutate(
#     Sex = ifelse(Sex == 'male', 1, 0)
# )
test <- test %>% mutate(
    Sex = mapvalues(Sex, from = c('male', 'female'), to = c('male' = 1, 'female' = 0))
)


print(head(train))
print(head(test))


train_embarked <- model.matrix(~ Embarked -1, data=train)
test_embarked <- model.matrix(~ Embarked -1, data = test)

colnames(train_embarked) <- gsub('Embarked', '', colnames(train_embarked))
colnames(test_embarked) <- str_replace_all(colnames(test_embarked), 'Embarked', '')

print(head(train_embarked))
print(head(test_embarked))

train <- cbind(train, train_embarked)
train <- subset(train, select = -c(PassengerId, Embarked))
test <- cbind(test, test_embarked)
test <- dplyr::select(test, -Embarked, -PassengerId)

train$title <- str_extract(train$Name, "\\b[A-Za-z]+(?=\\.)")
test$title <- str_extract(test$Name, "\\b[A-Za-z]+(?=\\.)")
# test$title <- str_extract(test$Name, "\\b[A-Za-z]+\\.?(?=\\s)") #JIKA MR.

# title_mapping <- c(
#     "^Mr$" = '0', "^Miss$" = '1', "^Mrs$" = '2',
#     "^Master$" = '3', "^Dr$" = '3', "^Rev$" = '3', "^Col$" = '3', "^Major$" = '3', "^Mlle$" = '3', "^Countess$" = '3',
#     "^Ms$" = '3', "^Lady$" = '3', "^Jonkheer$" = '3', "^Don$" = '3', "^Dona$" = '3', "^Mme$" = '3', "^Capt$" = '3', "^Sir$" = '3'
# )

title_mapping <- c(
    "Mr" = 0, "Miss" = 1, "Mrs" = 2,
    "Master" = 3, "Dr" = 3, "Rev" = 3, "Col" = 3, "Major" = 3, "Mlle" = 3, "Countess" = 3,
    "Ms" = 3, "Lady" = 3, "Jonkheer" = 3, "Don" = 3, "Dona" = 3, "Mme" = 3, "Capt" = 3, "Sir" = 3
)


train$title <- recode(train$title, !!!title_mapping, .default = NULL)
test$title <- mapvalues(test$title, from = names(title_mapping), to = title_mapping)
# train$title <- as.numeric(str_replace_all(train$title, title_mapping)) #TIDAK BISA KARENA TUJUAN REPLACEMENT YANG 0, 1, 1 BUKAN  '0', '1', '2'
# test$title <- as.numeric(str_replace_all(test$title, title_mapping)) #TIDAK BISA KARENA TUJUAN REPLACEMENT YANG 0, 1, 1 BUKAN  '0', '1', '2'

train <- dplyr::select(train, -c(Name, Ticket))
test <- subset(test, select = -c(Name, Ticket))

print(head(train))
print(head(test))
# print(colSums(is.na(train)))
# print(colSums(is.na(test)))

# train_data <- subset(train, select = -Survived)
# target <- factor(train$Survived)

# ctrl <- trainControl(method="cv", number=5)
# model <- train(train_data, target, method="rf", trControl=ctrl)
# print('accuracy mean')
# print(model$result)
# print(mean(model$result$Accuracy))

# pred <- predict(model, test)
# print(head(pred))

# df_test <- read.csv("ML/youtube/test.csv", na.strings = c("", "NA", " ", "N/A", "NULL"))
# hasil <- data.frame(
#     PassangerID = df_test$PassengerId,
#     survived = pred
# )
# print(head(hasil))