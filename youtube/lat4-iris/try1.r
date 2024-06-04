library(dplyr)
library(ggplot2)
library(stringr)
library(gridExtra)
library(RMySQL)
library(caTools)
library(dplyr)
library(tidyr)
library(plyr)
library(lubridate)
library(randomForest)
library(caret)
library(patchwork)
library(nnet)
library(klaR)  # untuk Naive Bayes
library(rpart) # untuk Decision Trees

df = read.csv('ML/youtube/lat4-iris/iris_data.csv', header = FALSE, col.names = c('Sepal_Length', 'Sepal_Width',  'Petal_Length',  'Petal_Width',  'Species'))
print(head(df))
print(str(df))
print(summary(df))
print(colSums(is.na(df)))


# p1 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = Sepal_Length), color = "blue") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p2 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = Sepal_Width), color = "green") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p3 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = Petal_Length), color = "orange") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal() 
# p4 <- ggplot(df) +
#     geom_point(aes(x = rownames(df), y = Petal_Width), color = "red") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()

# # # Menggabungkan plot menggunakan patchwork
# plot <- p1+p2+p3+p4
# # plot <- (p1 | p2) / (p3 | p4)
# print(plot)



# q1 <- quantile(df$Sepal_Length, 0.25)
# q3 <- quantile(df$Sepal_Length, 0.75)
# iqr <- q3 - q1
# # Mengaplikasikan aturan IQR untuk menyaring data
# df <- subset(df, Sepal_Length >= q1 - 1.5 * iqr & Sepal_Length <= q3 + 1.5 * iqr)



# Buat plot boxplot untuk setiap variabel
boxplot1 <- ggplot(df, aes(x = "", y = `Sepal_Length`)) +
    geom_boxplot() +
    theme_minimal() +
    labs(x = NULL)
boxplot2 <- ggplot(df, aes(x = "", y = `Sepal_Width`)) +
    geom_boxplot() +
    theme_minimal() +
    labs(x = NULL)
boxplot3 <- ggplot(df, aes(x = "", y = `Petal_Length`)) +
    geom_boxplot() +
    theme_minimal() +
    labs(x = NULL)
boxplot4 <- ggplot(df, aes(x = "", y = `Petal_Width`)) +
    geom_boxplot() +
    theme_minimal() +
    labs(x = NULL)
# Menggabungkan plot menggunakan grid.arrange
plot <- grid.arrange(boxplot1, boxplot2, boxplot3, boxplot4, ncol = 2)
# plot <- boxplot1 + boxplot2+boxplot3+boxplot4
print(plot)









print(unique(df$Species))
print(distinct(df, Species))
# df$Species <- case_when(
#     df$Species == 'Iris-setosa' ~ 0,
#     df$Species == 'Iris-versicolor' ~ 1,
#     df$Species == 'Iris-virginica' ~ 2,
#     .default = NULL
# )
df <- df %>% mutate(
    Species = case_when(
        Species == 'Iris-setosa' ~ 0,
        Species == 'Iris-versicolor' ~ 1,
        Species == 'Iris-virginica' ~ 2,
        .default = NULL
))
# df$Species <- ifelse(df$Species == 'Iris-setosa', 0, ifelse(df$Species == 'Iris-versicolor', 1, 2))
print(head(df))

# FOR REGRESSION ONLY
# x <- df[, 1:4]
# y <- df[, 5]


x <- dplyr::select(df, -Species)
y <- as.factor(df$Species)
# x <- subset(df, select = -Species)
# y <- as.factor(df$Species)
# x <- df[, 1:4]
# y <- as.factor(df$Species)
# x <- df[, 1:4]
# y <- factor(df$Species)


set.seed(42)
index <- createDataPartition(y, p = 0.7, list = TRUE)
print(head(index))


x_train <- x[index$Resample1, ]
x_test <- x[-index$Resample1, ]
y_train <- y[index$Resample1]
y_test <- y[-index$Resample1]



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



print(head(x_train, 10))
print(head(x_test))

cat('\n')

ctrl <- trainControl(method="cv", number=5)
model <- train(x_train, y_train, method="rf", trControl=ctrl)
print('accuracy mean')
print(model$result)
print(mean(model$result$Accuracy))
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
y_pred <- predict(model, x_test)

df_pred <- data.frame(
    y_test = y_test,
    y_pred = y_pred
)
print(df_pred)

df_pred <- filter(df_pred, y_test != y_pred)
print(df_pred)



# Calculate accuracy
accuracy <- confusionMatrix(data = y_pred, reference = y_test)$overall['Accuracy']
print(accuracy)


cat('\n\n\n\n')
# Calculate confusion matrix
print('Confusion Matrix')
cm <- confusionMatrix(data = y_pred, reference = y_test)
print(cm)
