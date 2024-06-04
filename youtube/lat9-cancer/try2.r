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


df <- read.csv('ML/youtube/lat9-cancer/data.csv')
print(head(df))
print(str(df))
print(summary(df))
print(colSums(is.na(df)))

df <- dplyr::select(df, -id, -X)

for (col in colnames(df)){
    print(col)
    print(head(unique(df[[col]])))
    cat('\n\n')
}



print(sapply(df, function(x) length(unique(na.omit(x)))))
print(sapply(df, function(x) length(unique(x[!is.na(x)]))))

# # TIDAK BISA KARENA ADANYA TABRAKAN DENGAN PACKAGE LAINNTA
# library(dplyr)
# library(tidyverse)
# library(ggplot2)
# hasil <- df %>% summarise(across(everything(), ~n_distinct(na.omit(.))))
# print(hasil)
# hasil <- df %>% summarise(across(everything(), ~n_distinct(.[!is.na(.)])))
# print(hasil)


print(unique(df$diagnosis))

# df$diagnosis <- ifelse(df$diagnosis == 'M', 0, 1)
df$diagnosis <- case_when(
    df$diagnosis == 'M' ~ 0,
    .default = 1
)
print(distinct(df, diagnosis))


# Membuat plot untuk setiap kolom
p1 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = diagnosis), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p2 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = radius_mean), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p3 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = texture_mean), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p4 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = perimeter_mean), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p5 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = area_mean), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p6 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = smoothness_mean), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p7 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = compactness_mean), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p8 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = concavity_mean), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p9 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = concave.points_mean), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p10 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = symmetry_mean), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p11 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = fractal_dimension_mean), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p12 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = radius_se), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p13 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = texture_se), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p14 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = perimeter_se), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p15 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = area_se), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p16 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = smoothness_se), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p17 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = compactness_se), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p18 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = concavity_se), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p19 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = concave.points_se), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p20 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = symmetry_se), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p21 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = fractal_dimension_se), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p22 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = radius_worst), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p23 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = texture_worst), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p24 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = perimeter_worst), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p25 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = area_worst), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p26 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = smoothness_worst), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p27 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = compactness_worst), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p28 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = concavity_worst), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p29 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = concave.points_worst), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p30 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = symmetry_worst), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p31 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = fractal_dimension_worst), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()

# Menggabungkan plot menggunakan patchwork
plot <- p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 + 
        p11 + p12 + p13 + p14 + p15 + p16 + p17 + p18 + p19 + p20 + 
        p21 + p22 + p23 + p24 + p25 + p26 + p27 + p28 + p29 + p30 + p31
print(plot)


remove_outlier <- function(df, col){
    # q1 <- quantile(df[[col]], 0.25)
    # q3 <- quantile(df[[col]], 0.75)
    # iqr <- q3 - q1
    # # Mengaplikasikan aturan IQR untuk menyaring data
    # df <- subset(df, df[[col]] >= q1 - 1.5 * iqr & df[[col]] <= q3 + 1.5 * iqr)


    # q1 <- quantile(df[[col]], 0.25)
    # q3 <- quantile(df[[col]], 0.75)
    # iqr <- q3 - q1
    # # Mengaplikasikan aturan IQR untuk menyaring data
    # df <- filter(df, df[[col]] >= q1 - 1.5 * iqr & df[[col]] <= q3 + 1.5 * iqr)


    q1 <- quantile(df[[col]], 0.25)
    q3 <- quantile(df[[col]], 0.75)
    iqr <- q3 - q1
    # Mengaplikasikan aturan IQR untuk menyaring data
    df <- filter(df, df[[col]] >= q1 - 1.5 * iqr, df[[col]] <= q3 + 1.5 * iqr)
    return(df)
}

for (col in colnames(df)){
    df <- remove_outlier(df, col)
}


# Membuat plot untuk setiap kolom
p1 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = diagnosis), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p2 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = radius_mean), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p3 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = texture_mean), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p4 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = perimeter_mean), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p5 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = area_mean), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p6 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = smoothness_mean), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p7 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = compactness_mean), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p8 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = concavity_mean), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p9 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = concave.points_mean), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p10 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = symmetry_mean), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p11 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = fractal_dimension_mean), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p12 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = radius_se), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p13 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = texture_se), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p14 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = perimeter_se), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p15 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = area_se), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p16 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = smoothness_se), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p17 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = compactness_se), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p18 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = concavity_se), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p19 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = concave.points_se), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p20 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = symmetry_se), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p21 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = fractal_dimension_se), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p22 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = radius_worst), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p23 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = texture_worst), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p24 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = perimeter_worst), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p25 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = area_worst), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p26 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = smoothness_worst), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p27 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = compactness_worst), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p28 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = concavity_worst), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p29 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = concave.points_worst), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p30 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = symmetry_worst), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p31 <- ggplot(df) +
    geom_point(aes(x = rownames(df), y = fractal_dimension_worst), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()

# Menggabungkan plot menggunakan patchwork
plot <- p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 + 
        p11 + p12 + p13 + p14 + p15 + p16 + p17 + p18 + p19 + p20 + 
        p21 + p22 + p23 + p24 + p25 + p26 + p27 + p28 + p29 + p30 + p31
print(plot)

print(head(df))
numvars <- colnames(dplyr::select(df, -diagnosis))
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


x_train <- subset(df_train, select = -diagnosis)
y_train <- as.factor(df_train$diagnosis)


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


support <- hasil$Features[hasil$Support == TRUE & hasil$Rank > 5]
not_support <- hasil$Features[hasil$Support == FALSE | hasil$Rank < 5]
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



x_test <- dplyr::select(df_test, -diagnosis)
x_test_rfe <- x_test[support]
y_test <- as.factor(df_test$diagnosis)

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


