library(dplyr)
library(ggplot2)
library(stringr)
library(gridExtra)
library(RMySQL)
library(dplyr)
library(zoo)
library(pROC)
library(ROCR)
library(tidyr)
library(plyr)
library(lubridate)
library(randomForest)
library(caret)
library(patchwork)
library(nnet)
library(klaR) # untuk Naive Bayes
library(rpart) # untuk Decision Trees


loan_data <- read.csv("ML/youtube/lat2-loan/lc_2016_2017.csv", na.strings = c("", "NA", " ", "N/A", "NULL"))
print(head(loan_data))
print(dim(loan_data))
print(colSums(is.na(loan_data)))

# loan_data$good_bad <- case_when(
#     loan_data$loan_status %in% c('Charged Off', 'Default', 'Late (16-30 days)', 'Late (31-120 days)') ~ 1,
#     .default = 0
# )

loan_data$good_bad <- ifelse(
    loan_data$loan_status %in% c("Charged Off", "Default", "Late (16-30 days)", "Late (31-120 days)"), 1, 0
)

print(head(loan_data))
print(table(loan_data$good_bad))
print(prop.table(table(loan_data$good_bad)))

# null_data <- data.frame(null = colSums(is.na(loan_data)) / dim(loan_data)[1] * 100)
# null_data <- data.frame(null = colSums(is.na(loan_data)) / nrow(loan_data) * 100)
null_data <- data.frame(null = colMeans(is.na(loan_data)) * 100)
null_data <- null_data %>%
    filter(null > 50) %>%
    arrange(desc(null))
print(null_data)

loan_data <- loan_data[, colMeans(is.na(loan_data)) < 0.5]

# null_data <- data.frame(null = colSums(is.na(loan_data)) / dim(loan_data)[1] * 100)
null_data <- data.frame(null = colSums(is.na(loan_data)) / nrow(loan_data) * 100)
# null_data <- data.frame(null = colMeans(is.na(loan_data)) * 100)
null_data <- null_data %>%
    filter(null > 50) %>%
    arrange(desc(null))
print(null_data)

x <- dplyr::select(loan_data, -good_bad)
y <- as.factor(loan_data$good_bad)
set.seed(42)
index <- createDataPartition(y, p = 0.8, list = TRUE)

x_train <- x[index$Resample1, ]
x_test <- x[-index$Resample1, ]
y_train <- y[index$Resample1]
y_test <- y[-index$Resample1]

print(prop.table(table(y_train)))
print(prop.table(table(y_test)))

for (col in colnames(x_train[, sapply(x_train, function(x) is.character(x) | is.logical(x))])) {
    print(col)
    print(head(x_train[[col]]))
    cat("\n\n\n")
}

col_need_to_clean <- c("term", "emp_length", "issue_d", "earliest_cr_line", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d")
x_train$term <- as.numeric(str_replace_all(x_train$term, " months", ""))

x_train$emp_length <- str_replace_all(x_train$emp_length, "\\+ years", "")
x_train$emp_length <- str_replace_all(x_train$emp_length, "< 1 year", "0")
x_train$emp_length <- str_replace_all(x_train$emp_length, " years", "")
x_train$emp_length <- str_replace_all(x_train$emp_length, " year", "")
x_train$emp_length[is.na(x_train$emp_length)] <- 0
x_train$emp_length <- as.numeric(x_train$emp_length)

penanggalan <- c("issue_d", "earliest_cr_line", "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d")
for (tanggal in penanggalan) {
    x_train[[tanggal]] <- my(x_train[[tanggal]])
}

print(head(x_train[col_need_to_clean]))



x_test$term <- as.numeric(str_replace_all(x_test$term, " months", ""))

x_test$emp_length <- str_replace_all(x_test$emp_length, "\\+ years", "")
x_test$emp_length <- str_replace_all(x_test$emp_length, "< 1 year", "0")
x_test$emp_length <- str_replace_all(x_test$emp_length, " years", "")
x_test$emp_length <- str_replace_all(x_test$emp_length, " year", "")
x_test$emp_length[is.na(x_test$emp_length)] <- 0
x_test$emp_length <- as.numeric(x_test$emp_length)


for (tanggal in penanggalan) {
    x_test[[tanggal]] <- my(x_test[[tanggal]])
}

print(head(x_test[col_need_to_clean]))


x_train <- x_train[col_need_to_clean]
x_train <- dplyr::select(x_train, -next_pymnt_d)
x_test <- x_test[col_need_to_clean]
x_test <- subset(x_test, select = -next_pymnt_d)

print(head(x_train))
print(head(x_test))

date_columns <- function(df, col) {
    today <- ymd(today())
    df[[col]] <- ymd(df[[col]])
    df[paste("mth since-", col)] <- round(as.numeric((today - df[[col]]) / 30))
    df <- dplyr::select(df, -col)
    # df <- subset(df, select = -col)
    return(df)
}

x_train <- date_columns(x_train, "issue_d")
x_train <- date_columns(x_train, "earliest_cr_line")
x_train <- date_columns(x_train, "last_pymnt_d")
x_train <- date_columns(x_train, "last_credit_pull_d")

x_test <- date_columns(x_test, "issue_d")
x_test <- date_columns(x_test, "earliest_cr_line")
x_test <- date_columns(x_test, "last_pymnt_d")
x_test <- date_columns(x_test, "last_credit_pull_d")

x_train <- na.aggregate(x_train, FUN = median) 
x_test <- na.aggregate(x_test, FUN = median) 

print(head(x_train))
print(head(x_test))
print(dim(x_train))
print(dim(x_test))
print(colSums(is.na(x_train)))
print(colSums(is.na(x_test)))



ctrl <- trainControl(method = "cv", number = 5)
# modelrf <- train(x_train, y_train, method = "rf", trControl = ctrl)
# print("accuracy mean")
# print(modelrf$result)
# print(mean(modelrf$result$Accuracy))
# cat("\n")

# modelnnet <- train(x_train, y_train, method="nnet", trControl=ctrl)
# print('accuracy mean')
# print(modelnnet$result)
# print(mean(modelnnet$result$Accuracy))
# cat('\n')

modelrpart <- train(x_train, y_train, method="rpart", trControl=ctrl)
print('accuracy mean')
print(modelrpart$result)
print(mean(modelrpart$result$Accuracy))
cat('\n')

# modelnb <- train(x_train, y_train, method="nb", trControl=ctrl)
# print('accuracy mean')
# print(modelnb$result)
# print(mean(modelnb$result$Accuracy))
# cat('\n')

y_pred <- predict(modelrpart, x_test)
print(head(y_pred))


df_pred <- data.frame(
    y_test = y_test,
    y_pred = y_pred
)
print(head(df_pred))

hasil <- df_pred[df_pred$y_test != df_pred$y_pred, ]
# hasil <- filter(df_pred, y_test != y_pred)
print(head(hasil))

accuracy <- confusionMatrix(data = y_pred, reference = y_test)$overall["Accuracy"]
print(accuracy)


y_pred <- predict(modelrpart, x_test, type = 'prob')
print(head(y_pred))


print('bagus')
y_pred <- predict(modelrpart, x_test, type = "prob")[,1]
print(head(y_pred))
print(head(y_pred > 0.5))
print(head(as.integer(y_pred > 0.5)))
print('jelek')
y_pred <- predict(modelrpart, x_test, type = "prob")[,2]
print(head(y_pred))
print(head(y_pred > 0.5))
print(head(as.integer(y_pred > 0.5)))


# Menghitung kurva ROC
roc_data <- roc(y_test, y_pred)

# Mendapatkan nilai FPR, TPR, dan thresholds
fpr <- roc_data$specificities
tpr <- roc_data$sensitivities
thresholds <- roc_data$thresholds

# Menampilkan hasil
print("FPR:")
print(fpr)
print("\nTPR:")
print(tpr)
print("\nThresholds:")
print(thresholds)

# Menghitung Youden's J statistic
j <- tpr - fpr
best_ix <- which.max(j)
best_thresh <- thresholds[best_ix]


cat("\n\nYouden's J Statistic:")
print(j)
cat("\nIndex of Best Threshold:")
print(best_ix)
cat("\nBest Threshold:")
print(best_thresh)

y_pred <- predict(modelrpart, x_test, type = "prob")[,2]
y_pred <- as.factor(as.numeric(y_pred > 0.066))

df_pred <- data.frame(
    y_test = y_test,
    y_pred = y_pred
)
print(head(df_pred))

hasil <- df_pred[df_pred$y_test != df_pred$y_pred, ]
# hasil <- filter(df_pred, y_test != y_pred)
print(head(hasil))

accuracy <- confusionMatrix(data = y_pred, reference = y_test)$overall["Accuracy"]
print(accuracy)