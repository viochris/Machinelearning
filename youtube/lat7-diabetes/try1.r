library(dplyr)
library(ggplot2)
library(stringr)
library(gridExtra)
library(RMySQL)
library(caTools)
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


df <- read.csv("ML/youtube/lat7-diabetes/diabetes.csv")
print(head(df))
print(dim(df))
print(str(df))
print(summary(df))
print(colSums(is.na(df)))
print(sapply(df, class))
print(sapply(df, function(x) length(unique(na.omit(x)))))

## TIDAK BISA KARENA ADANYA TABRAKAN DENGAN PACKAGE LAINNTA
# library(dplyr)
# library(tidyverse)
# library(ggplot2)
# null1 <- df %>% summarise(across(everything(), ~n_distinct(.[!is.na(.)])))
# print(as.data.frame(null1))

for (col in colnames(df)){
    print(col)
    print(unique(df[[col]]))
    cat('\n\n')
}

# p1 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = Pregnancies), color = "blue") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p2 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = Glucose), color = "green") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p3 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = BloodPressure), color = "orange") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal() 
# p4 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = SkinThickness), color = "red") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p5 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = Insulin), color = "orange") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal() 
# p6 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = BMI), color = "red") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p7 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = DiabetesPedigreeFunction), color = "orange") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal() 
# p8 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = Age), color = "red") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p9 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = Outcome), color = "red") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()

# # # Menggabungkan plot menggunakan patchwork
# plot <- p1+p2+p3+p4+p5+p6+p7+p8+p9
# # plot <- (p1 | p2) / (p3 | p4)
# print(plot)


# q1 <- quantile(df$Pregnancies, 0.25)
# q3 <- quantile(df$Pregnancies, 0.75)
# iqr <- q3 - q1
# # Mengaplikasikan aturan IQR untuk menyaring data
# df <- subset(df, Pregnancies >= q1 - 1.5 * iqr & Pregnancies <= q3 + 1.5 * iqr)

# q1 <- quantile(df$Glucose, 0.25)
# q3 <- quantile(df$Glucose, 0.75)
# iqr <- q3 - q1
# # Mengaplikasikan aturan IQR untuk menyaring data
# df <- subset(df, Glucose >= q1 - 1.5 * iqr & Glucose <= q3 + 1.5 * iqr)

# q1 <- quantile(df$BloodPressure , 0.25)
# q3 <- quantile(df$BloodPressure , 0.75)
# iqr <- q3 - q1
# # Mengaplikasikan aturan IQR untuk menyaring data
# df <- subset(df, BloodPressure  >= q1 - 1.5 * iqr & BloodPressure  <= q3 + 1.5 * iqr)

# q1 <- quantile(df$SkinThickness , 0.25)
# q3 <- quantile(df$SkinThickness , 0.75)
# iqr <- q3 - q1
# # Mengaplikasikan aturan IQR untuk menyaring data
# df <- subset(df, SkinThickness  >= q1 - 1.5 * iqr & SkinThickness  <= q3 + 1.5 * iqr)

# q1 <- quantile(df$Insulin  , 0.25)
# q3 <- quantile(df$Insulin  , 0.75)
# iqr <- q3 - q1
# # Mengaplikasikan aturan IQR untuk menyaring data
# df <- subset(df, Insulin   >= q1 - 1.5 * iqr & Insulin   <= q3 + 1.5 * iqr)

# q1 <- quantile(df$BMI , 0.25)
# q3 <- quantile(df$BMI , 0.75)
# iqr <- q3 - q1
# # Mengaplikasikan aturan IQR untuk menyaring data
# df <- subset(df, BMI  >= q1 - 1.5 * iqr & BMI  <= q3 + 1.5 * iqr)

# q1 <- quantile(df$DiabetesPedigreeFunction , 0.25)
# q3 <- quantile(df$DiabetesPedigreeFunction , 0.75)
# iqr <- q3 - q1
# # Mengaplikasikan aturan IQR untuk menyaring data
# df <- subset(df, DiabetesPedigreeFunction  >= q1 - 1.5 * iqr & DiabetesPedigreeFunction  <= q3 + 1.5 * iqr)

# q1 <- quantile(df$Age, 0.25)
# q3 <- quantile(df$Age, 0.75)
# iqr <- q3 - q1
# # Mengaplikasikan aturan IQR untuk menyaring data
# df <- subset(df, Age >= q1 - 1.5 * iqr & Age <= q3 + 1.5 * iqr)

# q1 <- quantile(df$Outcome , 0.25)
# q3 <- quantile(df$Outcome , 0.75)
# iqr <- q3 - q1
# # Mengaplikasikan aturan IQR untuk menyaring data
# df <- subset(df, Outcome  >= q1 - 1.5 * iqr & Outcome  <= q3 + 1.5 * iqr)

# p1 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = Pregnancies), color = "blue") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p2 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = Glucose), color = "green") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p3 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = BloodPressure), color = "orange") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal() 
# p4 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = SkinThickness), color = "red") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p5 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = Insulin), color = "orange") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal() 
# p6 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = BMI), color = "red") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p7 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = DiabetesPedigreeFunction), color = "orange") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal() 
# p8 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = Age), color = "red") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p9 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = Outcome), color = "red") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()

# # # Menggabungkan plot menggunakan patchwork
# plot <- p1+p2+p3+p4+p5+p6+p7+p8+p9
# # plot <- (p1 | p2) / (p3 | p4)
# print(plot)


# print(head(df))

# numvars <- c('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age')
# scaler <- preProcess(df[numvars], method = "range")
# df[numvars] <- predict(scaler, df[numvars])
# print(head(df))

# x <- subset(df, select = -Outcome)
# y <- as.factor(df$Outcome)

# set.seed(42)
# index <- createDataPartition(y, p = 0.7, list = TRUE)
# print(head(index))


# x_train <- x[index$Resample1, ]
# x_test <- x[-index$Resample1, ]
# y_train <- y[index$Resample1]
# y_test <- y[-index$Resample1]


# # split <- sample.split(y, SplitRatio = 0.7)
# # # x_train <- x[split, ]
# # # x_test <- x[!split, ]
# # # y_train <- y[split]
# # # y_test <- y[!split]

# # # Membuat data pelatihan
# # x_train <- subset(x, split == TRUE)
# # y_train <- subset(y, split == TRUE)
# # # Membuat data pengujian
# # x_test <- subset(x, split == FALSE)
# # y_test <- subset(y, split == FALSE)



# ctrl <- trainControl(method="cv", number=5)
# modelrf <- train(x_train, y_train, method="rf", trControl=ctrl)
# print('accuracy mean')
# print(modelrf$result)
# print(mean(modelrf$result$Accuracy))
# cat('\n')

# model <- train(x_train, y_train, method="nnet", trControl=ctrl)
# print('accuracy mean')
# print(model$result)
# print(mean(model$result$Accuracy))
# cat('\n')

# model <- train(x_train, y_train, method="rpart", trControl=ctrl)
# print('accuracy mean')
# print(model$result)
# print(mean(model$result$Accuracy))
# cat('\n')

# model <- train(x_train, y_train, method="nb", trControl=ctrl)
# print('accuracy mean')
# print(model$result)
# print(mean(model$result$Accuracy))
# cat('\n')

# y_pred_train <- predict(modelrf, x_train)
# df_pred <- data.frame(
#     y_train = y_train,
#     y_pred_train = y_pred_train
# )
# print(head(df_pred))

# df_pred <- filter(df_pred, y_train != y_pred_train)
# print(head(df_pred))

# # Calculate accuracy
# accuracy <- confusionMatrix(data = y_pred_train, reference = y_train)$overall['Accuracy']
# print(accuracy)

# cat('\n\n\n\n')
# # Calculate confusion matrix
# print('Confusion Matrix')
# cm <- confusionMatrix(data = y_pred_train, reference = y_train)
# print(cm)




# y_pred_test <- predict(modelrf, x_test)
# df_pred <- data.frame(
#     y_test = y_test,
#     y_pred_test = y_pred_test
# )
# print(head(df_pred))

# df_pred <- filter(df_pred, y_test != y_pred_test)
# print(head(df_pred))

# # Calculate accuracy
# accuracy <- confusionMatrix(data = y_pred_test, reference = y_test)$overall['Accuracy']
# print(accuracy)

# cat('\n\n\n\n')
# # Calculate confusion matrix
# print('Confusion Matrix')
# cm <- confusionMatrix(data = y_pred_test, reference = y_test)
# print(cm)
