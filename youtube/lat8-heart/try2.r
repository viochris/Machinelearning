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




df <- read.csv("ML/youtube/lat8-heart/heart.csv")
print(head(df))
print(dim(df))
print(str(df))
print(summary(df))
print(colSums(is.na(df)))
print(sapply(df, class))
print(sapply(df, function(x) length(unique(x[!is.na(x)]))))


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


## Set random seed
set.seed(42)
## Create train and test indices
split <- sample.split(df, SplitRatio = 0.7)

## Create train and test dataframes
df_train <- df[split,]
df_test <- df[!split,]
# df_train <- subset(df, split == TRUE)
# df_test <- subset(df, split == FALSE)

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

x_train <- subset(df_train, select = -target)
y_train <- as.factor(df_train$target)


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

support <- hasil$Features[hasil$Support == TRUE & hasil$Rank > 20]
not_support <- hasil$Features[hasil$Support == FALSE | hasil$Rank < 20]
print(support)
print(not_support)

x_train_rfe <- x_train[support]
print(head(x_train_rfe))

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

x_test <- subset(df_test, select = -target)
x_test_rfe <- x_test[support]
y_test <- as.factor(df_test$target)


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


