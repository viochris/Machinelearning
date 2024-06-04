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
# null1 <- df %>% summarise(across(everything(), n_distinct))
# print(as.data.frame(null1))



p1 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = Pregnancies), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p2 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = Glucose), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p3 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = BloodPressure), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal() 
p4 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = SkinThickness), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p5 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = Insulin), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal() 
p6 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = BMI), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p7 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = DiabetesPedigreeFunction), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal() 
p8 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = Age), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p9 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = Outcome), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()

# # Menggabungkan plot menggunakan patchwork
plot <- p1+p2+p3+p4+p5+p6+p7+p8+p9
# plot <- (p1 | p2) / (p3 | p4)
print(plot)


q1 <- quantile(df$Pregnancies, 0.25)
q3 <- quantile(df$Pregnancies, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, Pregnancies >= q1 - 1.5 * iqr & Pregnancies <= q3 + 1.5 * iqr)

q1 <- quantile(df$Glucose, 0.25)
q3 <- quantile(df$Glucose, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, Glucose >= q1 - 1.5 * iqr & Glucose <= q3 + 1.5 * iqr)

q1 <- quantile(df$BloodPressure , 0.25)
q3 <- quantile(df$BloodPressure , 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, BloodPressure  >= q1 - 1.5 * iqr & BloodPressure  <= q3 + 1.5 * iqr)

q1 <- quantile(df$SkinThickness , 0.25)
q3 <- quantile(df$SkinThickness , 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, SkinThickness  >= q1 - 1.5 * iqr & SkinThickness  <= q3 + 1.5 * iqr)

q1 <- quantile(df$Insulin  , 0.25)
q3 <- quantile(df$Insulin  , 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, Insulin   >= q1 - 1.5 * iqr & Insulin   <= q3 + 1.5 * iqr)

q1 <- quantile(df$BMI , 0.25)
q3 <- quantile(df$BMI , 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, BMI  >= q1 - 1.5 * iqr & BMI  <= q3 + 1.5 * iqr)

q1 <- quantile(df$DiabetesPedigreeFunction , 0.25)
q3 <- quantile(df$DiabetesPedigreeFunction , 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, DiabetesPedigreeFunction  >= q1 - 1.5 * iqr & DiabetesPedigreeFunction  <= q3 + 1.5 * iqr)

q1 <- quantile(df$Age, 0.25)
q3 <- quantile(df$Age, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, Age >= q1 - 1.5 * iqr & Age <= q3 + 1.5 * iqr)

q1 <- quantile(df$Outcome , 0.25)
q3 <- quantile(df$Outcome , 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
df <- subset(df, Outcome  >= q1 - 1.5 * iqr & Outcome  <= q3 + 1.5 * iqr)

p1 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = Pregnancies), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p2 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = Glucose), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p3 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = BloodPressure), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal() 
p4 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = SkinThickness), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p5 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = Insulin), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal() 
p6 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = BMI), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p7 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = DiabetesPedigreeFunction), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal() 
p8 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = Age), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p9 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = Outcome), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()

# # Menggabungkan plot menggunakan patchwork
plot <- p1+p2+p3+p4+p5+p6+p7+p8+p9
# plot <- (p1 | p2) / (p3 | p4)
print(plot)


print(head(df))

numvars <- c('Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age')
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

x_train <- subset(df_train, select = -Outcome)
y_train <- as.factor(df_train$Outcome)


ctrl <- trainControl(method="cv", number=5)
modelrf <- train(x_train, y_train, method="rf", trControl=ctrl)
print('accuracy mean')
print(modelrf$result)
print(mean(modelrf$result$Accuracy))
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


x_test <- subset(df_test, select = -Outcome)
x_test_rfe <- x_test[support]
y_test <- as.factor(df_test$Outcome)


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


