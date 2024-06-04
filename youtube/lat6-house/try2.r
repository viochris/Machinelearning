library(dplyr)
library(ggplot2)
library(stringr)
library(purrr)
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

train <- read.csv("ML/youtube/lat6-house/train.csv")
test <- read.csv("ML/youtube/lat6-house/test.csv")
print(head(train))
print(head(test))
print(dim(train))
print(dim(test))
print(colSums(is.na(train)))
print(colSums(is.na(test)))


# for(col in colnames(train)){
#     print(col)
#     print(head(train[[col]]))
#     print(head(unique(train[[col]])))
#     cat('\n\n\n')
# }


# p1 <- ggplot(train) +
#     geom_point(aes(x = rownames(train), y = SQUARE_FT), color = "blue") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p2 <- ggplot(train) +
#     geom_point(aes(x = rownames(train), y = BHK_NO.), color = "green") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p3 <- ggplot(test) +
#     geom_point(aes(x = rownames(test), y = SQUARE_FT), color = "orange") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p4 <- ggplot(test) +
#     geom_point(aes(x = rownames(test), y = BHK_NO.), color = "red") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p5 <- ggplot(train) +
#     geom_point(aes(x = rownames(train), y = TARGET.PRICE_IN_LACS.), color = "blue") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# # # Menggabungkan plot menggunakan patchwork
# plot <- p1 + p2 + p3 + p4 + p5 
# # plot <- (p1 | p2) / (p3 | p4|p5)
# print(plot)


q1 <- quantile(train$SQUARE_FT, 0.25)
q3 <- quantile(train$SQUARE_FT, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
train <- subset(train, SQUARE_FT >= q1 - 1.5 * iqr & SQUARE_FT <= q3 + 1.5 * iqr)

q1 <- quantile(train$BHK_NO., 0.25)
q3 <- quantile(train$BHK_NO., 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
# train <- filter(train, BHK_NO. >= q1 - 1.5 * iqr & BHK_NO. <= q3 + 1.5 * iqr)
train <- filter(train, BHK_NO. >= q1 - 1.5 * iqr, BHK_NO. <= q3 + 1.5 * iqr)

q1 <- quantile(test$SQUARE_FT, 0.25)
q3 <- quantile(test$SQUARE_FT, 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
test <- subset(test, SQUARE_FT >= q1 - 1.5 * iqr & SQUARE_FT <= q3 + 1.5 * iqr)

q1 <- quantile(test$BHK_NO., 0.25)
q3 <- quantile(test$BHK_NO., 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
test <- filter(test, BHK_NO. >= q1 - 1.5 * iqr & BHK_NO. <= q3 + 1.5 * iqr)
# test <- filter(test, BHK_NO. >= q1 - 1.5 * iqr, BHK_NO. <= q3 + 1.5 * iqr)

q1 <- quantile(train$TARGET.PRICE_IN_LACS., 0.25)
q3 <- quantile(train$TARGET.PRICE_IN_LACS., 0.75)
iqr <- q3 - q1
# Mengaplikasikan aturan IQR untuk menyaring data
train <- subset(train, TARGET.PRICE_IN_LACS. >= q1 - 1.5 * iqr & TARGET.PRICE_IN_LACS. <= q3 + 1.5 * iqr)


# p1 <- ggplot(train) +
#     geom_point(aes(x = rownames(train), y = SQUARE_FT), color = "blue") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p2 <- ggplot(train) +
#     geom_point(aes(x = rownames(train), y = BHK_NO.), color = "green") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p3 <- ggplot(test) +
#     geom_point(aes(x = rownames(test), y = SQUARE_FT), color = "orange") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p4 <- ggplot(test) +
#     geom_point(aes(x = rownames(test), y = BHK_NO.), color = "red") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# p5 <- ggplot(train) +
#     geom_point(aes(x = rownames(train), y = TARGET.PRICE_IN_LACS.), color = "blue") +
#     facet_grid(. ~ ., scales = "free") +
#     theme_minimal()
# # # Menggabungkan plot menggunakan patchwork
# # plot <- p1 + p2 + p3 + p4 + p5 
# plot <- (p1 | p2) / (p3 | p4|p5)
# print(plot)


for(col in colnames(train)){
    print(col)
    print(head(unique(train[[col]])))
    cat('\n\n\n')
}

train_posted <- model.matrix(~ POSTED_BY -1, data = train)
colnames(train_posted) <- gsub('POSTED_BY', '', colnames(train_posted))
train <- cbind(train, train_posted)
train <- subset(train, select = -POSTED_BY)
print(head(train))

test <- dummy_cols(test, select_columns = 'POSTED_BY')
colnames(test) <- str_replace_all(colnames(test), 'POSTED_BY_', '')
test <- dplyr::select(test, -POSTED_BY)
print(head(test))

train$BHK_OR_RK <- case_when(
    train$BHK_OR_RK == 'BHK' ~ 0,
    train$BHK_OR_RK == 'RK' ~ 1,
    .default = NULL
)

test$BHK_OR_RK <- ifelse(test$BHK_OR_RK == 'BHK', 0, 1)


train$alamat <- sapply(str_split(train$ADDRESS, ','), function(x) tail(x, 1))
test$alamat <- sapply(strsplit(test$ADDRESS, ','), function(x) tail(x, 1))

# train$alamat <- str_split(train$ADDRESS, ',') %>% map_chr(last)
# test$alamat <- strsplit(test$ADDRESS, ',') %>% map_chr(last)

train$alamat <- case_when(
    train$alamat %in% c('Bangalore', 'Mysore', 'Chennai', 'Mumbai', 'Delhi', 'Kolkata') ~ 0,
    train$alamat %in% c('Pune', 'Hyderabad', 'Ahmedabad', 'Surat', 'Jaipur') ~ 1,
    .default = 2
)

test$alamat <- case_when(
    test$alamat %in% c('Bangalore', 'Mysore', 'Chennai', 'Mumbai', 'Delhi', 'Kolkata') ~ 0,
    test$alamat %in% c('Pune', 'Hyderabad', 'Ahmedabad', 'Surat', 'Jaipur') ~ 1,
    .default = 2
)

train <- subset(train, select = -ADDRESS)
test <- dplyr::select(test, -ADDRESS)

numvars <- c('SQUARE_FT', 'TARGET.PRICE_IN_LACS.')
scaler <- preProcess(train[numvars], method = "range")
train[numvars] <- predict(scaler, train[numvars])
print(head(train))

print(head(train))
print(head(test))
print(sapply(train, class))
print(sapply(test, class))


## Set random seed
set.seed(42)
## Create train and test indices
split <- sample.split(train, SplitRatio = 0.7)

## Create train and test dataframes
# df_train <- train[split,]
# df_test <- train[!split,]
df_train <- subset(train, split == TRUE)
df_test <- subset(train, split == FALSE)

## Print first few rows of train and test dataframes
print(head(df_train))
print(head(df_test))
print(dim(train))
print(dim(df_train))
print(dim(df_test))

corr <- cor(df_train)
print(corr)
corplot <- corrplot(corr, method = 'color')
print(corplot)

x_train <- subset(df_train, select = -TARGET.PRICE_IN_LACS.)
y_train <- df_train$TARGET.PRICE_IN_LACS.


set.seed(42)
ctrl <- trainControl(method="cv", number=5)
lm_model <- train(x = x_train, y = y_train, method="lm", trControl=ctrl, metric = "Rsquared")
# lm_model <- train(x = x_train, y = y_train, method="lm", trControl=ctrl)
# lm_model <- train(x = train_data, y = target, method="lm", trControl=ctrl, metric = "Rsquared", seeds = list(seed = sample.int(1000, 5)))
print('accuracy mean')
print(lm_model$result)
print(lm_model$results$Rsquared)
print(mean(lm_model$results$Rsquared))
cat('\n\n')


glm_model <- train(x = x_train, y = y_train, method="glm", trControl=ctrl, metric = "Rsquared")
print('accuracy mean')
print(glm_model$result)
print(glm_model$results$Rsquared)
print(mean(glm_model$results$Rsquared))
cat('\n\n')


glmnet_model <- train(x = x_train, y = y_train, method="glmnet", trControl=ctrl, metric = "Rsquared")
print('accuracy mean')
print(glmnet_model$result)
print(glmnet_model$results$Rsquared)
print(mean(glmnet_model$results$Rsquared))
cat('\n\n')


# rf_model <- train(x = x_train, y = y_train, method="rf", trControl=ctrl, metric = "Rsquared")
# print('accuracy mean')
# print(rf_model$result)
# print(rf_model$results$Rsquared)
# print(mean(rf_model$results$Rsquared))
# cat('\n\n')



# Membuat kontrol RFE
# control <- rfeControl(functions = leapsFuncs, method = "cv", number =10)
# control <- rfeControl(functions = rfFuncs, method = "cv", number =10)
control <- rfeControl(functions = lmFuncs, method = "cv", number = 10)

# Menjalankan RFE
rfe_model <- rfe(x = x_train, y = y_train, sizes = c(6), rfeControl = control)
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



# support <- hasil$Features[hasil$Support == TRUE & hasil$Rank > 5]
# not_support <- hasil$Features[hasil$Support == FALSE | hasil$Rank < 5]
support <- hasil$Features[hasil$Support == TRUE & hasil$Rank > 0.05]
not_support <- hasil$Features[hasil$Support == FALSE | hasil$Rank < 0.05]
print(support)
print(not_support)

x_train_rfe <- x_train[support]
print(head(x_train_rfe))


rf_model <- train(x = x_train_rfe, y = y_train, method="lm", trControl=ctrl, metric = "Rsquared")
print('accuracy mean')
print(rf_model$result)
print(rf_model$results$Rsquared)
print(mean(rf_model$results$Rsquared))
cat('\n\n')


# Menghitung VIF
# Gabungkan x_train_rfe dan y_train menjadi satu data frame
data_train <- cbind(x_train_rfe, response = y_train)
# Membuat formula model linier dengan semua fitur dalam x_train_rfe
formula <- as.formula(paste("response ~", paste(names(x_train_rfe), collapse = " + ")))
# Membuat model linier menggunakan formula tersebut
model <- lm(formula, data = data_train)
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


y_train_price <- predict(rf_model, x_train_rfe)
print(head(y_train_price))

res <- y_train_price - y_train
# Membuat data frame untuk analisis residu
df <- data.frame(
    y_train = y_train,
    y_train_price = y_train_price,
    sisa = res
)
print(head(df))

# Menghitung R-squared
r_squared <- R2(y_train_price, y_train)
# Menampilkan hasil R-squared
print(r_squared)





x_test <- dplyr::select(df_test, -TARGET.PRICE_IN_LACS.)
x_test_rfe <- x_test[support]
y_test <- df_test$TARGET.PRICE_IN_LACS.

print(head(x_test_rfe))
print(head(y_test))

y_pred <- predict(rf_model, x_test_rfe)
print(head(y_pred))

res <- y_pred - y_test
# Membuat data frame untuk analisis residu
df <- data.frame(
    y_test = y_test,
    y_pred = y_pred,
    sisa = res
)
print(head(df))

# Menghitung R-squared
r_squared <- R2(y_pred, y_test)
# Menampilkan hasil R-squared
print(r_squared)


test <- test[support]
y_pred <- predict(rf_model, test)
print(head(y_pred))

