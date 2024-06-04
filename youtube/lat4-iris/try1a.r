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
library(klaR) # untuk Naive Bayes
library(rpart) # untuk Decision Trees

df <- read.csv("ML/youtube/lat4-iris/iris_data.csv", header = FALSE, col.names = c("Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)", "Species"))
print(head(df))
print(str(df))
print(summary(df))
print(colSums(is.na(df)))

print(unique(df$Species))
print(distinct(df, Species))
# df$Species <- case_when(
#     df$Species == 'Iris-setosa' ~ 0,
#     df$Species == 'Iris-versicolor' ~ 1,
#     df$Species == 'Iris-virginica' ~ 2,
#     .default = NULL
# )
df$Species <- ifelse(df$Species == "Iris-setosa", 0, ifelse(df$Species == "Iris-versicolor", 1, 2))
print(head(df))

# FOR REGRESSION
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
index <- createDataPartition(y, p = 0.7, list = FALSE)
print(head(index))


x_train <- x[index, ]
x_test <- x[-index, ]
y_train <- y[index]
y_test <- y[-index]


print(head(x_train, 10))
print(head(x_test))

cat("\n")

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


y_pred <- predict(model, x_train)

df_pred <- data.frame(
    y_train = y_train,
    y_pred = y_pred
)
print(df_pred)

df_pred <- filter(df_pred, y_train != y_pred)
print(df_pred)



# Calculate accuracy
accuracy <- confusionMatrix(data = y_pred, reference = y_train)$overall['Accuracy']
print(accuracy)


cat('\n\n\n\n')
# Calculate confusion matrix
print('Confusion Matrix')
cm <- confusionMatrix(data = y_pred, reference = y_train)
print(cm)




cat('\n\n\n\n\n')
print('yang x_test')
cat('\n\n\n\n\n')


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
