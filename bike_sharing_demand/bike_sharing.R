library(dplyr)
library(ggplot2)
library(gridExtra)
library(lubridate)
library(fasttime)

train <- read.csv("./train.csv")
test <- read.csv(("./test.csv"))

names(test)
names(train)
str(train)
summary(train)

test$casual <- -1
test$registered <- -1
test$count <- -1

bikes <- rbind(train, test)
str(bikes)
summary(bikes)

# bikes visualizaton
par(mfrow=c(4,2))
par(mar = rep(2, 4))
hist(bikes$temp)
hist(bikes$atemp)
hist(bikes$humidity)
hist(bikes$windspeed)
hist(bikes$weather)
hist(bikes$season)
hist(bikes$holiday)
hist(bikes$workingday)

prop.table(table(bikes$weather))
sum(bikes$casual)
sum(bikes$registered)

# look for NA values
sum(is.na(bikes))

# create individual columns for day, month, year, weekday, weekend
bikes %>%
  mutate(datetime = fastPOSIXct(datetime, "GMT")) %>%
  mutate(month = month(datetime), 
         year = year(datetime), 
         day = day(datetime), 
         hour = hour(datetime),
         weekday = wday(datetime)) %>%
  mutate(weekend = as.integer(weekday %in% c(1,7))) -> bikes

# encode categorical variables accordingly
bikes$season <- as.factor(bikes$season)
bikes$holiday <- as.factor(bikes$holiday)
bikes$workingday <- as.factor(bikes$workingday)
bikes$weather <- as.factor(bikes$weather)
bikes$log_cas <- log1p(bikes$casual) 
bikes$log_reg <- log1p(bikes$registered)


train <- bikes %>%
  filter(day < 20)

test <- bikes %>% 
  filter(day > 19)

# bike demandand by hour
ggplot(train, aes(factor(hour), count)) + geom_boxplot()
ggplot(train, aes(factor(hour), log1p(count))) + geom_boxplot()
# bike demand by registered and casual
regs <- ggplot(train, aes(factor(hour), registered)) + geom_boxplot()
cas <- ggplot(train, aes(factor(hour), casual)) + geom_boxplot()
grid.arrange(regs, cas, nrow = 2)

# total bike demand 2011 vs 2012
ggplot(train, aes(year, count)) + geom_boxplot()
t.test(count ~ year, data = train)
ggplot(train, aes(factor(year), count)) + geom_boxplot()
# bike demand by casual
ggplot(train, aes(factor(year), registered)) + geom_boxplot()
t.test(registered ~ year, data = train)

# wind vs count
ggplot(train, aes(windspeed, log1p(count))) + geom_point()

# temp and atemp vis
par(mfrow = c(2, 1))
hist(bikes$temp)
hist(bikes$atemp)

# correlation between numeric variables
sub <- data.frame(train$registered, train$casual, train$count, 
                  train$temp, train$atemp, train$humidity)
cor(sub)

# use average of atemp and temp
bikes %>% 
  mutate(aveg_temp = (atemp + temp)/2) -> bikes

# bike demand by day
by_day_reg <- ggplot(train, aes(weekday, registered)) + geom_boxplot()
by_day_cas <- ggplot(train, aes(weekday, casual)) + geom_boxplot()
grid.arrange(by_day_reg, by_day_cas, nrow=2)

# feature creation
str(bikes)

train <- bikes %>%
  filter(day < 20)

test <- bikes %>% 
  filter(day > 19)


library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

# create hour buckets for registered users using results from decision tree.
dt <- rpart(registered ~ hour, data = train)
fancyRpartPlot(dt)

bikes$reg_hr <- 0
bikes$reg_hr[bikes$hour < 8] <- 1
bikes$reg_hr[bikes$hour >= 22] <- 2
bikes$reg_hr[bikes$hour > 9 & bikes$hour < 18] <- 3
bikes$reg_hr[bikes$hour == 8] <- 4
bikes$reg_hr[bikes$hour == 9] <- 5
bikes$reg_hr[bikes$hour == 20 | bikes$hour == 21] <- 6
bikes$reg_hr[bikes$hour == 19 | bikes$hour == 18] <- 7
summary(bikes$reg_hr)

# create hour buckets for casual users using results from decision tree.
dt <- rpart(casual ~ hour, data = train)
fancyRpartPlot(dt)

bikes$cas_hr <- 0
bikes$cas_hr[bikes$hour < 8] <- 1
bikes$cas_hr[bikes$hour >= 20] <- 2
bikes$cas_hr[bikes$hour >= 8 & bikes$hour < 20] <- 3
summary(bikes$cas_hr)
ggplot(train, aes(factor(cas_hr), casual)) + geom_boxplot()

# create hour buckets for count 

dt <- rpart(count ~ hour, data = train)
fancyRpartPlot(dt)
# create temp buckets for registered using decision tree
dt <- rpart(registered ~ aveg_temp, data = train)
fancyRpartPlot(dt)

bikes$temp_bin <- 0
bikes$temp_bin[bikes$aveg_temp < 14] <- 1
bikes$temp_bin[bikes$aveg_temp >= 14 & bikes$aveg_temp < 28] <- 2
bikes$temp_bin[bikes$aveg_temp >= 28] <- 3
summary(bikes$temp_bin)
ggplot(train, aes(factor(temp_bin), casual)) + geom_boxplot()

# create year bins since demand increased from 2011-2012

bikes$year_bin[bikes$year=='2011']=1
bikes$year_bin[bikes$year=='2011' & bikes$month>3]=2
bikes$year_bin[bikes$year=='2011' & bikes$month>6]=3
bikes$year_bin[bikes$year=='2011' & bikes$month>9]=4
bikes$year_bin[bikes$year=='2012']=5
bikes$year_bin[bikes$year=='2012' & bikes$month>3]=6
bikes$year_bin[bikes$year=='2012' & bikes$month>6]=7
bikes$year_bin[bikes$year=='2012' & bikes$month>9]=8
table(bikes$year_bin)

# look at bikes 
str(bikes)
bikes$hour <- as.factor(bikes$hour)
bikes$month <- as.factor(bikes$month)
bikes$year <- as.factor(bikes$year)
bikes$weekday <- as.factor(bikes$weekday)
bikes$day <- as.factor(bikes$day)

# subset train and test data
train <- bikes %>%
  filter(count > 0)
str(train)

test <- bikes %>% 
  filter(count < 0) %>%
  select(-log_reg, -log_cas,-casual, -registered, -count)
str(test)

# model training 
library(xgboost)
library(Ckmeans.1d.dp)
require(Matrix)
require(data.table)

# features for registered users
features_reg <- c('season', 'holiday', 'workingday', 'weather', 'aveg_temp', 
                  'humidity', 'windspeed', 'month',  'year', 'hour', 'day',
                  'weekday', 'weekend', 'reg_hr', 'temp_bin', 'year_bin')
# subsetting bikes dataframe with features of interest.
X_train_reg <- data.table(train[, features_reg])

# one hot encoding for categorical variables
sparseM_reg <- sparse.model.matrix(~.-1, X_train_reg)

# train labels
y_train_reg <- train$log_reg

# train model for registered users
bst_reg <- xgboost(data = sparseM_reg, label = y_train_reg, max_depth = 5, 
               eta = 0.1, subsample = 0.9, nrounds = 150)
# plot feature importance
importance <- xgb.importance(feature_names = colnames(sparseM_reg), model = bst)

# prepare test data for predictions and make predictions
test_x <- data.table(test[, features_reg])
test_sparseM <- sparse.model.matrix(~.-1, test_x)

preds_reg <- predict(bst_reg, test_sparseM)
preds_reg <- expm1(preds_reg)

# features for casual users
features_cas <- c('season', 'holiday', 'workingday', 'weather', 'aveg_temp', 
                  'humidity', 'windspeed', 'month',  'year', 'hour', 'day',
                  'weekday', 'weekend', 'cas_hr', 'temp_bin', 'year_bin')

X_train_cas <- data.table(train[, features_cas])
sparseM_cas <- sparse.model.matrix(~.-1, X_train_cas)
y_train_cas <- train$log_cas 
bst_cas <- xgboost(data = sparseM_cas, label = y_train_cas, max_depth = 5, 
                   eta = 0.1, subsample = 0.9, nrounds = 150)
importance_cas <- xgb.importance(feature_names = colnames(sparseM_cas), model = bst)

test_x_cas <- data.table(test[, features_cas])
test_sparseM_cas <- sparse.model.matrix(~.-1, test_x_cas)

preds_cas <- predict(bst_cas, test_sparseM_cas)
preds_cas <- expm1(preds_cas)

count <- preds_cas + preds_reg
subs <- data.frame(datetime = test$datetime, count = count)
write.csv(subs, file = "submissions.csv", row.names = FALSE)

# test xgboost with count as response
# create log of count variable
bikes$log_count <- log1p(bikes$count)

# subset bike df with features of interest
train <- bikes %>%
  filter(count > 0) %>%
  select(-datetime, -temp, -atemp, -casual, -registered, -cas_hr, -count, 
         -weekday)
# prepare training data
X_train <- train %>%
  select(-log_count) %>%
  as.matrix()

# train labels and model training
y_train <- train$log_count
dtrain <- xgb.DMatrix(X_train, label = y_train)
model <- xgb.train(data = dtrain, nround = 150, max_depth = 5, eta = 0.1, subsample = 0.9)
xgb.importance(feature_names = colnames(X_train), model) %>% xgb.plot.importance()
features <- names(train)

# prepare test data and making predictions
test <- bikes %>% 
  filter(count < 0) %>%
  select(-temp, -atemp, -casual, -registered, -cas_hr, -count, 
         -weekday, -log_count)

x_test <- test %>% 
  select(-datetime) %>%
  as.matrix()
str(test)

preds <- predict(model, x_test)
count <- expm1(preds)
subs <- data.frame(datetime = test$datetime, count = count)
write.csv(subs, file = "submissions.csv", row.names = FALSE)

# test xgboost with count as response w/o engineered features except for date vars

bikes$log_count <- log1p(bikes$count)

train <- bikes %>%
  filter(count > 0) %>%
  select(-datetime, -temp, -atemp, -casual, -registered, -count)
X_train <- train %>%
  select(-log_count) %>%
  as.matrix()

y_train <- train$log_count
dtrain <- xgb.DMatrix(X_train, label = y_train)
model <- xgboost(data = dtrain, nround = 150, max_depth = 6, eta = 0.1, subsample = 0.9)
xgb.importance(feature_names = colnames(X_train), model) %>% xgb.plot.importance()

test <- bikes %>% 
  filter(count < 0) %>%
  select(-temp, -atemp, -casual, -registered, -count, -log_count)

x_test <- test %>% 
  select(-datetime) %>%
  as.matrix()
str(test)

preds <- predict(model, x_test)
count <- expm1(preds)
subs <- data.frame(datetime = test$datetime, count = count)
write.csv(subs, file = "submissions2.csv", row.names = FALSE)
