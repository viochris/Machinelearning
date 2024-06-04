library(dplyr)
library(ggplot2)
library(stringr)
library(gridExtra)
library(RMySQL)
library(corrplot)
library(caTools)
library(dplyr)
library(fastDummies)
library(zoo)
library(pROC)
library(ROCR)
library(leaps)
library(car)
library(tidyr)
library(plyr)
library(lubridate)
library(randomForest)
library(caret)
library(patchwork)
library(nnet)
library(klaR) # untuk Naive Bayes
library(rpart) # untuk Decision Trees


df <- read.csv("ML/youtube/lat4-iris/iris_data.csv", header = FALSE, col.names = c("Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Species"))
print(head(df))
print(str(df))
print(summary(df))
print(colSums(is.na(df)))


p1 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = Sepal_Length), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p2 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = Sepal_Width), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p3 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = Petal_Length), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p4 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = Petal_Width), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()

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

cat("\n\n\n\n")
print(unique(df$Species))
# df$Species <- case_when(
#     df$Species == 'Iris-setosa' ~ 0,
#     df$Species == 'Iris-versicolor' ~ 1,
#     df$Species == 'Iris-virginica' ~ 2
# )
df$Species <- ifelse(df$Species == "Iris-setosa", 0, ifelse(df$Species == "Iris-versicolor", 1, 2))
print(head(df))

numvars <- c("Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width")
scaler <- preProcess(df[numvars], method = "range")
df[numvars] <- predict(scaler, df[numvars])
print(head(df))


## Set random seed
set.seed(42)
## Create train and test indices
split <- sample.split(df, SplitRatio = 0.7)

## Create train and test dataframes
# df_train <- df[split,]
# df_test <- df[!split,]
df_train <- subset(df, split == TRUE)
df_test <- subset(df, split == FALSE)

## Print first few rows of train and test dataframes
print(head(df_train))
print(head(df_test))
print(dim(df))
print(dim(df_train))
print(dim(df_test))


corr <- cor(df_train)
print(corr)
corplot <- corrplot(corr, method = 'color')
print(corplot)


x_train <- subset(df_train, select = -Species)
y_train <- as.factor(df_train$Species)
print(head(x_train))
print(head(y_train))


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


# Membuat kontrol RFE
# control <- rfeControl(functions = leapsFuncs, method = "cv", number =10)
control <- rfeControl(functions = rfFuncs, method = "cv", number =10)
# Menjalankan RFE
rfe_model <- rfe(x = x_train, y = y_train, sizes = c(3), rfeControl = control)
# Mendapatkan hasil support dan ranking
support <- predictors(rfe_model)
ranking <- varImp(rfe_model, scale = FALSE)

print(support)
print(ranking)

# Membuat dataframe hasil
hasil <- data.frame(
    Features = rownames(ranking),
    Support = rownames(ranking) %in% support,
    Rank = ranking$Overall
)

print(hasil)

support <- hasil$Features[hasil$Support == TRUE]
not_support <- hasil$Features[hasil$Support == FALSE]
print(support)
print(not_support)

x_train_rfe <- x_train[support]
print(head(x_train_rfe))

ctrl <- trainControl(method="cv", number=5)
modelrf <- train(x_train_rfe, y_train, method="rf", trControl=ctrl)
print('accuracy mean')
print(modelrf$result)
print(mean(modelrf$result$Accuracy))
cat('\n')


# Menghitung VIF
# Gabungkan x_train_rfe dan y_train menjadi satu data frame
data_train <- cbind(x_train_rfe, response = y_train)
# Membuat formula model linier dengan semua fitur dalam x_train_rfe
formula <- as.formula(paste("response ~", paste(names(x_train_rfe), collapse = " + ")))
# Membuat model linier menggunakan formula tersebut
model <- glm(formula, data = data_train, family = binomial)
# Menghitung VIF
vif_values <- vif(model)
print(vif_values)

# Membuat data frame dengan hasil VIF
vif_df <- data.frame(
    Features = names(vif_values),
    VIF = vif_values
)
vif_df <- vif_df[order(-vif_df$VIF), ]

print(vif_df)


y_pred <- predict(modelrf, x_train_rfe)
df_pred <- data.frame(
    y_train = y_train,
    y_pred = y_pred
)
print(head(df_pred))

df_pred <- filter(df_pred, y_train != y_pred)
print(head(df_pred))

# Calculate accuracy
accuracy <- confusionMatrix(data = y_pred, reference = y_train)$overall['Accuracy']
print(accuracy)

cat('\n\n\n\n')
# Calculate confusion matrix
print('Confusion Matrix')
cm <- confusionMatrix(data = y_pred, reference = y_train)
print(cm)


x_test <- dplyr::select(df_test, -Species)
x_test_rfe <- x_test[support]
y_test <- as.factor(df_test$Species)


y_pred <- predict(modelrf, x_test_rfe)
df_pred <- data.frame(
    y_test = y_test,
    y_pred = y_pred
)
print(head(df_pred))

df_pred <- filter(df_pred, y_test != y_pred)
print(head(df_pred))

# Calculate accuracy
accuracy <- confusionMatrix(data = y_pred, reference = y_test)$overall['Accuracy']
print(accuracy)

cat('\n\n\n\n')
# Calculate confusion matrix
print('Confusion Matrix')
cm <- confusionMatrix(data = y_pred, reference = y_test)
print(cm)


