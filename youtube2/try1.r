library(dplyr)
library(caret)

loan <- read.csv("Praktek/YouTube2/lc_2016_2017.csv", na.strings = c("", "NA", " ", "N/A", "NULL"))

print(head(loan))
print(str(loan))
print(summary(loan))
print(dim(loan))
print(colSums(is.na(loan)))
cat("\n\n")

print(unique(loan$loan_status))
loan$Good_Bad <- ifelse(
    loan$loan_status %in% c("Charged Off", "Default", "Late (16-30 days)", "Late (31-120 days)"),
    1, 0
)

# loan <- loan %>% mutate(
#     Good_Bad2 = case_when(
#         loan_status %in% c("Charged Off", "Default", "Late (16-30 days)", "Late (31-120 days)") ~ 1,
#         TRUE ~ 0
#     )
# )

print(head(loan))


missing_value <- as.data.frame(colMeans(is.na(loan)) * 100)
colnames(missing_value) <- "nullpct"
print(missing_value)
# missing_value <- as.data.frame((colSums(is.na(loan)) / nrow(loan)) * 100)
# colnames(missing_value) <- "nullpct"
# print(missing_value)
# missing_value <- as.data.frame((colSums(is.na(loan)) / dim(loan)[1]) * 100)
# colnames(missing_value) <- "nullpct"
# print(missing_value)

missing_value  <- missing_value %>% filter(nullpct > 50)
missing_value <- missing_value %>% arrange(desc(nullpct))
print(missing_value)


loan <- loan[, colMeans(is.na(loan)) <= 0.5]


missing_value  <- missing_value %>% filter(nullpct > 50) %>% arrange(desc(nullpct))
print(missing_value)









x <- subset(loan, select= -Good_Bad)
# y <- loan$Good_Bad #coock untuk regresi
y <- factor(loan$Good_Bad) #cocok untuk klasifikasi

print(head(x))
print(head(y))

set.seed(42)
index <- createDataPartition(y, p = 0.8, list = FALSE)

# Memisahkan data menjadi set pelatihan dan pengujian
x_train <- x[index, ]
x_test <- x[-index, ]
y_train <- y[index]
y_test <- y[-index]


print(table(y_train))
print(table(y_test))
# Mencetak proporsi nilai dalam y_train
print(prop.table(table(y_train)))
# Mencetak proporsi nilai dalam y_test
print(prop.table(table(y_test)))
