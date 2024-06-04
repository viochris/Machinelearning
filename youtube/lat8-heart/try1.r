library(dplyr)
library(ggplot2)
library(stringr)
library(gridExtra)
library(RMySQL)
library(dplyr)
library(caTools)
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



df <- read.csv("ML/youtube/lat8-heart/heart.csv")
print(head(df))
print(dim(df))
print(str(df))
print(summary(df))
print(colSums(is.na(df)))
print(sapply(df, class))
print(sapply(df, function(x) length(unique(x[!is.na(x)]))))


## TIDAK BISA KARENA ADANYA TABRAKAN DENGAN PACKAGE LAINNTA
# library(dplyr)
# library(tidyverse)
# library(ggplot2)
# null1 <- df %>% summarise(across(everything(), ~n_distinct(na.omit(.))))
# print(as.data.frame(null1))


for (col in colnames(df)){
    print(col)
    print(unique(df[[col]]))
    cat('\n\n')
}


p1 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = age), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p2 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = sex), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p3 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = cp), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p4 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = trestbps), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal() 
p5 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = chol), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p6 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = fbs), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal() 
p7 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = restecg), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p8 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = thalach), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal() 
p9 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = exang), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p10 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = oldpeak), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p11 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = slope), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p12 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = ca), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p13 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = thal), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p14 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = target), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()

# # Menggabungkan plot menggunakan patchwork
plot <- p1+p2+p3+p4+p5+p6+p7+p8+p9+p10+p11+p12+p13+p14
# plot <- (p1 | p2) / (p3 | p4)
print(plot)



q1 <- quantile(df$age, 0.25)
q3 <- quantile(df$age, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, age >= q1 - 1.5 * iqr & age <= q3 + 1.5 * iqr)

q1 <- quantile(df$sex, 0.25)
q3 <- quantile(df$sex, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, sex >= q1 - 1.5 * iqr & sex <= q3 + 1.5 * iqr)

q1 <- quantile(df$cp , 0.25)
q3 <- quantile(df$cp , 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, cp  >= q1 - 1.5 * iqr & cp  <= q3 + 1.5 * iqr)

q1 <- quantile(df$trestbps , 0.25)
q3 <- quantile(df$trestbps , 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, trestbps  >= q1 - 1.5 * iqr & trestbps  <= q3 + 1.5 * iqr)

q1 <- quantile(df$chol  , 0.25)
q3 <- quantile(df$chol  , 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, chol   >= q1 - 1.5 * iqr & chol   <= q3 + 1.5 * iqr)

q1 <- quantile(df$fbs , 0.25)
q3 <- quantile(df$fbs , 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, fbs  >= q1 - 1.5 * iqr & fbs  <= q3 + 1.5 * iqr)

q1 <- quantile(df$restecg , 0.25)
q3 <- quantile(df$restecg , 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, restecg  >= q1 - 1.5 * iqr & restecg  <= q3 + 1.5 * iqr)

q1 <- quantile(df$thalach, 0.25)
q3 <- quantile(df$thalach, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, thalach >= q1 - 1.5 * iqr & thalach <= q3 + 1.5 * iqr)

q1 <- quantile(df$exang , 0.25)
q3 <- quantile(df$exang , 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, exang  >= q1 - 1.5 * iqr & exang  <= q3 + 1.5 * iqr)

q1 <- quantile(df$oldpeak, 0.25)
q3 <- quantile(df$oldpeak, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, oldpeak >= q1 - 1.5 * iqr & oldpeak <= q3 + 1.5 * iqr)

q1 <- quantile(df$slope , 0.25)
q3 <- quantile(df$slope , 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, slope  >= q1 - 1.5 * iqr & slope  <= q3 + 1.5 * iqr)

q1 <- quantile(df$ca, 0.25)
q3 <- quantile(df$ca, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, ca >= q1 - 1.5 * iqr & ca <= q3 + 1.5 * iqr)

q1 <- quantile(df$thal , 0.25)
q3 <- quantile(df$thal , 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, thal  >= q1 - 1.5 * iqr & thal  <= q3 + 1.5 * iqr)

q1 <- quantile(df$target , 0.25)
q3 <- quantile(df$target , 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, target  >= q1 - 1.5 * iqr & target  <= q3 + 1.5 * iqr)



p1 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = age), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p2 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = sex), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p3 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = cp), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p4 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = trestbps), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal() 
p5 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = chol), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p6 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = fbs), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal() 
p7 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = restecg), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p8 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = thalach), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal() 
p9 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = exang), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p10 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = oldpeak), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p11 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = slope), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p12 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = ca), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p13 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = thal), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p14 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = target), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()

# # Menggabungkan plot menggunakan patchwork
plot <- p1+p2+p3+p4+p5+p6+p7+p8+p9+p10+p11+p12+p13+p14
# plot <- (p1 | p2) / (p3 | p4)
print(plot)


print(head(df))

numvars <- c('age', 'trestbps', 'chol', 'thalach', 'oldpeak')
scaler <- preProcess(df[numvars], method = "range")
df[numvars] <- predict(scaler, df[numvars])
print(head(df))

x <- dplyr::select(df, -target)
y <- as.factor(df$target)


set.seed(42)
# index <- createDataPartition(y, p = 0.7, list = TRUE)
# print(head(index))


# x_train <- x[index$Resample1, ]
# x_test <- x[-index$Resample1, ]
# y_train <- y[index$Resample1]
# y_test <- y[-index$Resample1]


split <- sample.split(y, SplitRatio = 0.7)
# x_train <- x[split, ]
# x_test <- x[!split, ]
# y_train <- y[split]
# y_test <- y[!split]

# Membuat data pelatihan
x_train <- subset(x, split == TRUE)
y_train <- subset(y, split == TRUE)
# Membuat data pengujian
x_test <- subset(x, split == FALSE)
y_test <- subset(y, split == FALSE)



ctrl <- trainControl(method="cv", number=5)
modelrf <- train(x_train, y_train, method="rf", trControl=ctrl)
print('accuracy mean')
print(modelrf$result)
print(mean(modelrf$result$Accuracy))
cat('\n')

model <- train(x_train, y_train, method="nnet", trControl=ctrl)
print('accuracy mean')
print(model$result)
print(mean(model$result$Accuracy))
cat('\n')

model <- train(x_train, y_train, method="rpart", trControl=ctrl)
print('accuracy mean')
print(model$result)
print(mean(model$result$Accuracy))
cat('\n')

model <- train(x_train, y_train, method="nb", trControl=ctrl)
print('accuracy mean')
print(model$result)
print(mean(model$result$Accuracy))
cat('\n')



y_pred_train <- predict(modelrf, x_train)
df_pred <- data.frame(
    y_train = y_train,
    y_pred_train = y_pred_train
)
print(head(df_pred))

df_pred <- filter(df_pred, y_train != y_pred_train)
print(head(df_pred))

# Calculate accuracy
accuracy <- confusionMatrix(data = y_pred_train, reference = y_train)$overall['Accuracy']
print(accuracy)

cat('\n\n\n\n')
# Calculate confusion matrix
print('Confusion Matrix')
cm <- confusionMatrix(data = y_pred_train, reference = y_train)
print(cm)




y_pred_test <- predict(modelrf, x_test)
df_pred <- data.frame(
    y_test = y_test,
    y_pred_test = y_pred_test
)
print(head(df_pred))

df_pred <- filter(df_pred, y_test != y_pred_test)
print(head(df_pred))

# Calculate accuracy
accuracy <- confusionMatrix(data = y_pred_test, reference = y_test)$overall['Accuracy']
print(accuracy)

cat('\n\n\n\n')
# Calculate confusion matrix
print('Confusion Matrix')
cm <- confusionMatrix(data = y_pred_test, reference = y_test)
print(cm)
