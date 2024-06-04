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


house <- read.csv('ML/youtube/lat3-housing/Housing.csv')
print(head(house))
print(colSums(is.na(house)))
print(colSums(is.na(house)) / nrow(house))
print(colSums(is.na(house)) / dim(house)[1])
print(colMeans(is.na(house)))
print(sapply(house, class))

p1 <- ggplot(house) +
    geom_point(aes(x = rownames(house), y = price), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p2 <- ggplot(house) +
    geom_point(aes(x = rownames(house), y = area), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p3 <- ggplot(house) +
    geom_point(aes(x = rownames(house), y = bedrooms), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal() 
p4 <- ggplot(house) +
    geom_point(aes(x = rownames(house), y = bathrooms), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p5 <- ggplot(house) +
    geom_point(aes(x = rownames(house), y = stories), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p6 <- ggplot(house) +
    geom_point(aes(x = rownames(house), y = parking), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()

# # Menggabungkan plot menggunakan patchwork
plot <- p1+p2+p3+p4+p5+p6
# plot <- (p1 | p2) / (p3 | p4)
print(plot)


q1 <- quantile(house$price, 0.25)
q3 <- quantile(house$price, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
house <- subset(house, price >= q1 - 1.5 * iqr & price <= q3 + 1.5 * iqr)


q1 <- quantile(house$area, 0.25)
q3 <- quantile(house$area, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
# house <- filter(house, area >= q1 - 1.5 * iqr & area <= q3 + 1.5 * iqr)
house <- filter(house, area >= q1 - 1.5 * iqr, area <= q3 + 1.5 * iqr)


p1 <- ggplot(house) +
    geom_point(aes(x = rownames(house), y = price), color = "blue") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p2 <- ggplot(house) +
    geom_point(aes(x = rownames(house), y = area), color = "green") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p3 <- ggplot(house) +
    geom_point(aes(x = rownames(house), y = bedrooms), color = "orange") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal() 
p4 <- ggplot(house) +
    geom_point(aes(x = rownames(house), y = bathrooms), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p5 <- ggplot(house) +
    geom_point(aes(x = rownames(house), y = stories), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()
p6 <- ggplot(house) +
    geom_point(aes(x = rownames(house), y = parking), color = "red") +
    facet_grid(. ~ ., scales = "free") +
    theme_minimal()

# # Menggabungkan plot menggunakan patchwork
# plot <- p1+p2+p3+p4+p5+p6
plot <- (p1 | p2 |p3) / (p4 | p5|p6)
print(plot)

varlist <- c('mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea')

# for (col in varlist){
#     house[[col]] <- ifelse(house[[col]] == 'yes', 1, 0)
# }

# ubah <- function(x){
#     hasil <- ifelse(x == 'yes', 1, 0)
#     return (hasil)
# }
# house[varlist] <- sapply(house[varlist], ubah)

ubah <- function(df, col){
    df[col] <- ifelse(df[col] == 'yes', 1, 0)
    return(df)
}
house <- ubah(house, varlist)


# ubah <- function(df, col){
#     df[col] <- ifelse(df[col] == 'yes', 1, 0)
# }
# house[varlist] <- ubah(house, varlist)


print(head(house))


# # # status <- model.matrix(~ furnishingstatus -1, data = house)
# # # colnames(status) <- gsub('furnishingstatus', '', colnames(status))
# # # # colnames(status) <- str_replace_all(colnames(status), 'furnishingstatus', '')
# # # house <- cbind(house, status)
# # # house <- subset(house, select = -c(furnished, furnishingstatus))
# # # # house <- dplyr::select(house, -c(furnished, furnishingstatus))
# # # # house <- dplyr::select(house, -furnished, -furnishingstatus)
# # # print(head(house))


# house <- dummy_cols(house, select_columns = 'furnishingstatus', remove_first_dummy = TRUE)
# # colnames(house) <- gsub('furnishingstatus_', '', colnames(house))
# colnames(house) <- str_replace_all(colnames(house), 'furnishingstatus_', '')
# # house <- subset(house, select = -furnishingstatus)
# house <- dplyr::select(house, -furnishingstatus)
# print(head(house))



# ## Set random seed
# set.seed(42)
# ## Create train and test indices
# split <- sample.split(house, SplitRatio = 0.7)

# ## Create train and test dataframes
# # df_train <- house[split,]
# # df_test <- house[!split,]
# df_train <- subset(house, split == TRUE)
# df_test <- subset(house, split == FALSE)

# ## Print first few rows of train and test dataframes
# print(head(df_train))
# print(head(df_test))
# print(dim(house))
# print(dim(df_train))
# print(dim(df_test))


# numvars <- c('area', 'bedrooms', 'bathrooms', 'stories', 'parking','price')
# scaler <- preProcess(df_train[numvars], method = "range")
# df_train[numvars] <- predict(scaler, df_train[numvars])
# print(head(df_train))


# corr <- cor(df_train)
# print(corr)
# corplot <- corrplot(corr, method = 'color')
# print(corplot)

# x_train <- subset(df_train, select = -price)
# y_train <- df_train$price
# print(head(x_train))
# print(head(y_train))




# set.seed(42)
# ctrl <- trainControl(method="cv", number=5)
# lm_model <- train(x = x_train, y = y_train, method="lm", trControl=ctrl, metric = "Rsquared")
# # lm_model <- train(x = x_train, y = y_train, method="lm", trControl=ctrl)
# # lm_model <- train(x = train_data, y = target, method="lm", trControl=ctrl, metric = "Rsquared", seeds = list(seed = sample.int(1000, 5)))
# print('accuracy mean')
# print(lm_model$result)
# print(lm_model$results$Rsquared)
# print(mean(lm_model$results$Rsquared))
# cat('\n\n')


# glm_model <- train(x = x_train, y = y_train, method="glm", trControl=ctrl, metric = "Rsquared")
# print('accuracy mean')
# print(glm_model$result)
# print(glm_model$results$Rsquared)
# print(mean(glm_model$results$Rsquared))
# cat('\n\n')


# glmnet_model <- train(x = x_train, y = y_train, method="glmnet", trControl=ctrl, metric = "Rsquared")
# print('accuracy mean')
# print(glmnet_model$result)
# print(glmnet_model$results$Rsquared)
# print(mean(glmnet_model$results$Rsquared))
# cat('\n\n')


# rf_model <- train(x = x_train, y = y_train, method="rf", trControl=ctrl, metric = "Rsquared")
# print('accuracy mean')
# print(rf_model$result)
# print(rf_model$results$Rsquared)
# print(mean(rf_model$results$Rsquared))
# cat('\n\n')



# # Membuat kontrol RFE
# # control <- rfeControl(functions = leapsFuncs, method = "cv", number =10)
# control <- rfeControl(functions = rfFuncs, method = "cv", number =10)
# # Menjalankan RFE
# rfe_model <- rfe(x = x_train, y = y_train, sizes = c(6), rfeControl = control)
# # Mendapatkan hasil support dan ranking
# support <- predictors(rfe_model)
# ranking <- varImp(rfe_model, scale = FALSE)

# print(support)
# print(ranking)

# # Membuat dataframe hasil
# hasil <- data.frame(
#     Features = rownames(ranking),
#     Support = rownames(ranking) %in% support,
#     Rank = ranking$Overall
# )

# print(hasil)

# support <- hasil$Features[hasil$Support == TRUE & hasil$Rank >= 15]
# not_support <- hasil$Features[hasil$Support == FALSE | hasil$Rank < 15]
# print(support)
# print(not_support)


# x_train_rfe <- x_train[support]
# print(head(x_train_rfe))


# rf_model <- train(x = x_train_rfe, y = y_train, method="rf", trControl=ctrl, metric = "Rsquared")
# print('accuracy mean')
# print(rf_model$result)
# print(rf_model$results$Rsquared)
# print(mean(rf_model$results$Rsquared))
# cat('\n\n')


# # Menghitung VIF
# # Gabungkan x_train_rfe dan y_train menjadi satu data frame
# data_train <- cbind(x_train_rfe, response = y_train)
# # Membuat formula model linier dengan semua fitur dalam x_train_rfe
# formula <- as.formula(paste("response ~", paste(names(x_train_rfe), collapse = " + ")))
# # Membuat model linier menggunakan formula tersebut
# model <- lm(formula, data = data_train)
# # Menghitung VIF
# vif_values <- vif(model)
# print(vif_values)

# # Membuat data frame dengan hasil VIF
# vif_df <- data.frame(
#     Features = names(vif_values),
#     VIF = vif_values
# )
# vif_df <- vif_df[order(-vif_df$VIF), ]

# print(vif_df)


# y_train_price <- predict(rf_model, x_train_rfe)
# print(head(y_train_price))

# res <- y_train_price - y_train
# # Membuat data frame untuk analisis residu
# df <- data.frame(
#     y_train = y_train,
#     y_train_price = y_train_price,
#     sisa = res
# )
# print(head(df))

# # Menghitung R-squared
# r_squared <- R2(y_train_price, y_train)
# # Menampilkan hasil R-squared
# print(r_squared)



# df_test[numvars] <- predict(scaler, df_test[numvars])
# print(head(df_test))

# x_test <- dplyr::select(df_test, -price)
# x_test_rfe <- x_test[support]
# y_test <- df_test$price

# print(head(x_test_rfe))
# print(head(y_test))

# y_pred <- predict(rf_model, x_test_rfe)
# print(head(y_pred))

# # Menghitung R-squared
# r_squared <- R2(y_pred, y_test)
# # Menampilkan hasil R-squared
# print(r_squared)