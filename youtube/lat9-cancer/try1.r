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


df <- read.csv('ML/youtube/lat9-cancer/data.csv')
print(head(df))
print(str(df))
print(summary(df))
print(colSums(is.na(df)))

df <- dplyr::select(df, -id, -X)

for (col in colnames(df)){
    print(col)
    print(head(unique(df[col])))
    cat('\n\n')
}



print(sapply(df, function(x) length(unique(na.omit(x)))))
# print(sapply(df, function(x) length(unique(x[!is.na(x)]))))


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


    q1 <- quantile(df[[col]], 0.25)
    q3 <- quantile(df[[col]], 0.75)
    iqr <- q3 - q1
    # Mengaplikasikan aturan IQR untuk menyaring data
    df <- filter(df, df[[col]] >= q1 - 1.5 * iqr & df[[col]] <= q3 + 1.5 * iqr)


    # q1 <- quantile(df[[col]], 0.25)
    # q3 <- quantile(df[[col]], 0.75)
    # iqr <- q3 - q1
    # # Mengaplikasikan aturan IQR untuk menyaring data
    # df <- filter(df, df[[col]] >= q1 - 1.5 * iqr, df[[col]] <= q3 + 1.5 * iqr)
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

x <- subset(df, select = -diagnosis)
y <- as.factor(df$diagnosis)



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
