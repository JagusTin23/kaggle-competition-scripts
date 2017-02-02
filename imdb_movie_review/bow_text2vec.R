library(caret)
library(text2vec)
library(glmnet)
library(doMC)

setwd("./documents/data-science-projects/kaggle_movie_reviews")
set.seed(100)

# Reading review parsed data.  
# Review data has been parsed using a Python script to remove all HTML and  
# markup tags.  

train_file <- "./data/train_data_parsed.csv"
parsed_trainD <- read.csv(train_file, header = TRUE, stringsAsFactors = FALSE)

# Remove review ID column.  
parsed_trainD <- parsed_trainD[, -1]
parsed_trainD$sentiment <- factor(parsed_trainD$sentiment)
str(parsed_trainD)
# Partitioning data into train and test datasets, use 80% for training model.  
inTrain <- createDataPartition(parsed_trainD$sentiment, p = 0.8, list = FALSE)
trainD <- parsed_trainD[inTrain, ]
testD <- parsed_trainD[-inTrain, ]

# Verify proportion of sentiment in both train and test sets
prop.table(table(trainD$sentiment))
prop.table(table(testD$sentiment))

# Performs text cleaning, lowercase text, etc.
cleanReviews <- function(input) {
  # convert to lowercase
  output <- tolower(input)
  # remove numbers
  output <- gsub("\\S*[0-9]+\\S*", " ", output)
  # remove punctuation, except intra-word apostrophe
  output <- gsub("[^[:alnum:][:space:]']", " ", output)
  output <- gsub("(\\w['-]\\w)|[[:punct:]]", "\\1", output)
  # compress and trim whitespace
  output <- gsub("\\s+"," ",output)
  output <- gsub("^\\s+|\\s+$", "", output)
  return(output)
}

train_clean <- cleanReviews(trainD$review_parsed)
test_clean <- cleanReviews(testD$review_parsed)
train_lables <- trainD$sentiment
test_lables <- testD$sentiment

rm(list = ls()[!ls() %in% c("test_clean", "test_lables", 
                            "train_clean", "train_lables")])


# Tokenizing train reviews, each word is a token.  
tokens <- train_clean %>%
  word_tokenizer()
#
it <- itoken(tokens) 

# Creates vocabulary for term, 1-grams.  
# Prune vocabulary based on frequency of terms.  
vocab <- create_vocabulary(it) %>%
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5, 
                   doc_proportion_min = 0.001)
# Vectorizing terms.  
vectorizer <- vocab_vectorizer(vocab)

# Create TF-IDF matrix, a weighted DTM.  
dtm_1gram <- tokens %>% 
  itoken() %>% 
  create_dtm(vectorizer) %>% 
  transform_tfidf()

dim(dtm_1gram)

registerDoMC(cores = 2)

model_1gram_lasso <- cv.glmnet(x = dtm_1gram, y = train_lables, 
                 family = 'binomial', 
                 alpha = 1,
                 type.measure = "auc",
                 nfolds = 5,
                 thresh = 1e-3,
                 maxit = 1e3, 
                 parallel = TRUE)

plot(model_1gram_lasso)
max(model_1gram_lasso$cvm)

model_1gram_Ridged <- cv.glmnet(x = dtm_1gram, y = train_lables, 
                       family = 'binomial', 
                       alpha = 0,
                       type.measure = "auc",
                       nfolds = 5,
                       thresh = 1e-3,
                       maxit = 1e3, 
                       parallel = TRUE)

plot(model_1gram_Ridged)
max(model_1gram_Ridged$cvm)
rm(dtm_1gram)
# Building model using 2-grams 
it <- itoken(tokens)

vocab2N <- create_vocabulary(it, ngram = c(1L, 2L)) %>% 
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5, 
                   doc_proportion_min = 0.0009)

vectorizer2N <- vocab_vectorizer(vocab2N)

dtm_2gram <- tokens %>% 
  itoken() %>% 
  create_dtm(vectorizer2N) %>% 
  transform_tfidf()

dim(dtm_2gram)
model_2gram <- cv.glmnet(x = dtm_2gram, y = train_lables, 
                 family = 'binomial', 
                 alpha = 1,
                 type.measure = "auc",
                 nfolds = 5,
                 thresh = 1e-3,
                 maxit = 1e3, 
                 parallel = TRUE)
plot(model_2gram)
max(model_2gram$cvm)

fit_2gram_ridged <- cv.glmnet(x = dtm_2gram, y = train_lables, 
                         family = 'binomial', 
                         alpha = 0,
                         type.measure = "auc",
                         nfolds = 5,
                         thresh = 1e-3,
                         maxit = 1e3,
                         parallel = TRUE)
plot(fit_2gram_ridged)
max(fit_2gram_ridged$cvm)
rm(dtm_2gram)
# Building model using 3-grams 
it <- itoken(tokens)

vocab3N <- create_vocabulary(it, ngram = c(1L, 3L)) %>% 
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5, 
                   doc_proportion_min = 0.0009)

vectorizer3N <- vocab_vectorizer(vocab3N)

dtm_3gram <- tokens %>% 
  itoken() %>% 
  create_dtm(vectorizer3N) %>% 
  transform_tfidf()

dim(dtm_3gram)
model_3gram_lasso <- cv.glmnet(x = dtm_3gram, y = train_lables, 
                         family = 'binomial', 
                         alpha = 1,
                         type.measure = "auc",
                         nfolds = 5,
                         thresh = 1e-3,
                         maxit = 1e3, 
                         parallel = TRUE)
plot(model_3gram_lasso)
max(model_3gram_lasso$cvm)

fit_3gram_ridged <- cv.glmnet(x = dtm_3gram, y = train_lables, 
                              family = 'binomial', 
                              alpha = 0,
                              type.measure = "auc",
                              nfolds = 5,
                              thresh = 1e-3,
                              maxit = 1e3,
                              parallel = TRUE)
plot(fit_3gram_ridged)
max(fit_3gram_ridged$cvm)
rm(dtm_3gram)

# Evaluate model on test data 
test_tokens <- test_clean %>%
  word_tokenizer()

# 1-gram test 
dtm_test_1gram <- test_tokens %>%
  itoken() %>%
  create_dtm(vectorizer) %>%
  transform_tf()
dim(dtm_test_1gram)

# Prediction lasso model 
preds_1gram_lasso <- predict(model_1gram_lasso, newx = dtm_test_1gram,
                             type = "class", s = "lambda.min")
lasso_1gram_acc <- confusionMatrix(preds_1gram_lasso, test_lables)$overall[['Accuracy']]*100

# Predictions ridged model
preds_1gram_ridged <- predict(model_1gram_Ridged, newx = dtm_test_1gram,
                             type = "class", s = "lambda.min")
ridge_1gram_acc <- confusionMatrix(preds_1gram_ridged, test_lables)$overall[['Accuracy']]*100


# Measuring accurancy on model 2-grams
dtm_test_2gram <- test_tokens %>%
  itoken() %>%
  create_dtm(vectorizer2N) %>%
  transform_tf()

preds_2gram_lasso <- predict(model_2gram, newx = dtm_test_2gram, 
                            type = "class", s = "lambda.min")
lasso_2gram_acc <- confusionMatrix(preds_2gram_lasso, test_lables)$overall[['Accuracy']]*100

preds_2gram_ridged <- predict(fit_2gram_ridged, newx = dtm_test_2gram, 
                             type = "class", s = "lambda.min")

ridge_2gram_acc <- confusionMatrix(preds_2gram_ridged, test_lables)$overall[['Accuracy']]*100

# Measuring accuracy for 3-grams
dtm_test_3gram <- test_tokens %>%
  itoken() %>%
  create_dtm(vectorizer3N) %>%
  transform_tf()

preds_3gram_lasso <- predict(model_3gram_lasso, newx = dtm_test_3gram, 
                             type = "class", s = "lambda.min")
lasso_3gram_acc <- confusionMatrix(preds_3gram_lasso, test_lables)$overall[['Accuracy']]*100

preds_3gram_ridged <- predict(fit_3gram_ridged, newx = dtm_test_3gram, 
                              type = "class", s = "lambda.min")

ridge_3gram_acc <- confusionMatrix(preds_3gram_ridged, test_lables)$overall[['Accuracy']]*100

# Showing results in data frame.  
model_type <- c("1-grams", "2-grams", "3-grams")
auc_lasso <- c(max(model_1gram_lasso$cvm), max(model_2gram$cvm),  max(model_3gram_lasso$cvm))
auc_lasso <- round(auc_lasso, 4)*100
auc_ridged <- c(max(model_1gram_Ridged$cvm), max(fit_2gram_ridged$cvm), max(fit_3gram_ridged$cvm))
auc_ridged <- round(auc_ridged, 4)*100
accuracy_lasso <- c(lasso_1gram_acc, lasso_2gram_acc, lasso_3gram_acc)
accuray_ridged <- c(ridge_1gram_acc, ridge_2gram_acc, ridge_3gram_acc) 
resultsDF <- data.frame(model_type, auc_lasso, auc_ridged, accuracy_lasso, accuray_ridged)

resultsDF

head(preds_3gram_ridged)

