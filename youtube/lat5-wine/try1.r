library(dplyr)
library(ggplot2)
library(stringr)
library(gridExtra)
library(RMySQL)
library(dplyr)
library(tidyr)
library(plyr)
library(lubridate)
library(randomForest)
library(caret)
library(patchwork)
library(nnet)
library(klaR) # untuk Naive Bayes
library(rpart) # untuk Decision Trees

df <- read.csv("ML/youtube/lat5-wine/WineQT.csv")
print(head(df))
print(dim(df))
print(colSums(is.na(df)))
cat("\n\n")

df <- dplyr::select(df, -Id)
print(head(df))


# Buat plot boxplot untuk setiap variabel
boxplot1 <- ggplot(df, aes(x = "", y = `fixed.acidity`)) +
    geom_boxplot() +
    theme_minimal() +
    labs(x = NULL)
boxplot2 <- ggplot(df, aes(x = "", y = `volatile.acidity`)) +
    geom_boxplot() +
    theme_minimal() +
    labs(x = NULL)
boxplot3 <- ggplot(df, aes(x = "", y = `citric.acid`)) +
    geom_boxplot() +
    theme_minimal() +
    labs(x = NULL)
boxplot4 <- ggplot(df, aes(x = "", y = `residual.sugar`)) +
    geom_boxplot() +
    theme_minimal() +
    labs(x = NULL)
boxplot5 <- ggplot(df, aes(x = "", y = `chlorides`)) +
    geom_boxplot() +
    theme_minimal() +
    labs(x = NULL)
boxplot6 <- ggplot(df, aes(x = "", y = `free.sulfur.dioxide`)) +
    geom_boxplot() +
    theme_minimal() +
    labs(x = NULL)
boxplot7 <- ggplot(df, aes(x = "", y = `total.sulfur.dioxide`)) +
    geom_boxplot() +
    theme_minimal() +
    labs(x = NULL)
boxplot8 <- ggplot(df, aes(x = "", y = `density`)) +
    geom_boxplot() +
    theme_minimal() +
    labs(x = NULL)
boxplot9 <- ggplot(df, aes(x = "", y = `pH`)) +
    geom_boxplot() +
    theme_minimal() +
    labs(x = NULL)
boxplot10 <- ggplot(df, aes(x = "", y = `sulphates`)) +
    geom_boxplot() +
    theme_minimal() +
    labs(x = NULL)
boxplot11 <- ggplot(df, aes(x = "", y = `alcohol`)) +
    geom_boxplot() +
    theme_minimal() +
    labs(x = NULL)
boxplot12 <- ggplot(df, aes(x = "", y = `quality`)) +
    geom_boxplot() +
    theme_minimal() +
    labs(x = NULL)
# Menggabungkan plot menggunakan grid.arrange
# plot <- grid.arrange(boxplot1, boxplot2, boxplot3, boxplot4, ncol = 2)
plot <- boxplot1 + boxplot2 + boxplot3 + boxplot4 + boxplot5 + boxplot6 + boxplot7 + boxplot8 + boxplot9 + boxplot10 + boxplot11 + boxplot12
print(plot)



p1 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = fixed.acidity), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p2 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = volatile.acidity), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p3 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = citric.acid), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p4 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = residual.sugar), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p5 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = chlorides), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p6 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = free.sulfur.dioxide), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p7 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = total.sulfur.dioxide), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p8 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = density), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p9 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = pH), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p10 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = sulphates), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p11 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = alcohol), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p12 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = quality), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()

# # Menggabungkan plot menggunakan patchwork
plot <- p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 + p12
# plot <- (p1 | p2) / (p3 | p4)
print(plot)


q1 <- quantile(df$fixed.acidity, 0.25)
q3 <- quantile(df$fixed.acidity, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, fixed.acidity >= q1 - 1.5 * iqr & fixed.acidity <= q3 + 1.5 * iqr)

q1 <- quantile(df$volatile.acidity, 0.25)
q3 <- quantile(df$volatile.acidity, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
# df <- filter(df, volatile.acidity >= q1 - 1.5 * iqr & volatile.acidity <= q3 + 1.5 * iqr)
df <- filter(df, volatile.acidity >= q1 - 1.5 * iqr , volatile.acidity <= q3 + 1.5 * iqr)


q1 <- quantile(df$citric.acid, 0.25)
q3 <- quantile(df$citric.acid, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- df[(df$citric.acid >= q1 - 1.5 * iqr & df$citric.acid <= q3 + 1.5 * iqr),]

q1 <- quantile(df$residual.sugar, 0.25)
q3 <- quantile(df$residual.sugar, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, residual.sugar >= q1 - 1.5 * iqr & residual.sugar <= q3 + 1.5 * iqr)

q1 <- quantile(df$chlorides, 0.25)
q3 <- quantile(df$chlorides, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
# df <- filter(df, chlorides >= q1 - 1.5 * iqr & chlorides <= q3 + 1.5 * iqr)
df <- filter(df, chlorides >= q1 - 1.5 * iqr , chlorides <= q3 + 1.5 * iqr)


q1 <- quantile(df$free.sulfur.dioxide, 0.25)
q3 <- quantile(df$free.sulfur.dioxide, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
# df <- filter(df, free.sulfur.dioxide >= q1 - 1.5 * iqr & free.sulfur.dioxide <= q3 + 1.5 * iqr)
df <- filter(df, free.sulfur.dioxide >= q1 - 1.5 * iqr , free.sulfur.dioxide <= q3 + 1.5 * iqr)


q1 <- quantile(df$total.sulfur.dioxide, 0.25)
q3 <- quantile(df$total.sulfur.dioxide, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, total.sulfur.dioxide >= q1 - 1.5 * iqr & total.sulfur.dioxide <= q3 + 1.5 * iqr)


q1 <- quantile(df$density, 0.25)
q3 <- quantile(df$density, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, density >= q1 - 1.5 * iqr & density <= q3 + 1.5 * iqr)


q1 <- quantile(df$pH, 0.25)
q3 <- quantile(df$pH, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, pH >= q1 - 1.5 * iqr & pH <= q3 + 1.5 * iqr)


q1 <- quantile(df$sulphates, 0.25)
q3 <- quantile(df$sulphates, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, sulphates >= q1 - 1.5 * iqr & sulphates <= q3 + 1.5 * iqr)


q1 <- quantile(df$alcohol, 0.25)
q3 <- quantile(df$alcohol, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, alcohol >= q1 - 1.5 * iqr & alcohol <= q3 + 1.5 * iqr)


q1 <- quantile(df$quality, 0.25)
q3 <- quantile(df$quality, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, quality >= q1 - 1.5 * iqr & quality <= q3 + 1.5 * iqr)



p1 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = fixed.acidity), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p2 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = volatile.acidity), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p3 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = citric.acid), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p4 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = residual.sugar), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p5 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = chlorides), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p6 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = free.sulfur.dioxide), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p7 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = total.sulfur.dioxide), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p8 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = density), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p9 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = pH), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p10 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = sulphates), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p11 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = alcohol), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p12 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = quality), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()

# # Menggabungkan plot menggunakan patchwork
# plot <- p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 + p12
plot <- (p1 | p2 |p3) / (p4 | p5 | p6) / (p7|p8|p9) / (p10|p11|p12)
print(plot)

print(head(df))
print(dim(df))

cat('\n\n\n\n\n')

# Membuat objek scaler
scaler <- preProcess(dplyr::select(df, -quality), method = "range")
numvars <- colnames(dplyr::select(df, -quality))
df[numvars] <- predict(scaler, df[numvars])
print(head(df))

x <- dplyr::select(df, -quality)
y <- factor(df$quality)

set.seed(42)
index <- createDataPartition(y, p = 0.7, list = TRUE)
print(head(index))


x_train <- x[index$Resample1, ]
x_test <- x[-index$Resample1, ]
y_train <- y[index$Resample1]
y_test <- y[-index$Resample1]


ctrl <- trainControl(method = "cv", number = 5)
modelrf <- train(x_train, y_train, method = "rf", trControl = ctrl)
print("accuracy mean")
print(modelrf$result)
print(mean(modelrf$result$Accuracy))
cat("\n")

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

y_train_pred <- predict(modelrf, x_train)
print(y_train_pred)

df_pred <- data.frame(
    y_train = y_train,
    y_train_pred = y_train_pred
)
print(head(df_pred))

df_pred <- filter(df_pred, y_train != y_train_pred)
print(head(df_pred))
# Calculate accuracy
accuracy <- confusionMatrix(data = y_train_pred, reference = y_train)$overall["Accuracy"]
print(accuracy)

cat("\n\n\n\n")


y_test_pred <- predict(modelrf, x_test)
print(y_test_pred)

df_pred <- data.frame(
    y_test = y_test,
    y_test_pred = y_test_pred
)
print(head(df_pred))

df_pred <- filter(df_pred, y_test != y_test_pred)
print(head(df_pred))
# Calculate accuracy
accuracy <- confusionMatrix(data = y_test_pred, reference = y_test)$overall["Accuracy"]
print(accuracy)

cat("\n\n\n\n")
# Calculate confusion matrix
print("Confusion Matrix")
cm <- confusionMatrix(data = y_test_pred, reference = y_test)
print(cm)
