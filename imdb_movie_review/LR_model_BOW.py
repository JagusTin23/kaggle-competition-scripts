#!/usr/bin/env python

# train and predict, based on validation params

import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score as AUC


# test and train files
train_file = 'labeledTrainData.tsv' 
test_file = 'testData.tsv'
output_file = 'predictions_bow.csv'
# read files
train = pd.read_csv( train_file, header = 0, delimiter = '\t', quoting = 3 )
test = pd.read_csv( test_file, header = 0, delimiter = '\t', quoting = 3 )


# train_i, test_i = train_test_split( np.arange( len( data )), train_size = 0.8, random_state = 44 )

# train = data.ix[train_i]
# test = data.ix[test_i]

# function to remove html tags and non-letter characters.
def clean_text(raw_review):
	# parse html
	review_text = BeautifulSoup( raw_review, 'lxml' ).get_text()
	# remove non-letters
	review_text = re.sub('[^a-zA-Z]', ' ', review_text)
	# return cleaned text
	review_text = review_text.lower().split()
	return ' '.join( review_text )

# clean train reviews
clean_train_reviews = []

for review in train['review']:
	clean_train_reviews.append( clean_text( review ) )

# clean test reveiws
clean_test_reviews = []

for review in test['review']:
	clean_test_reviews.append( clean_text( review ) )

# create tfidf vectorizer
vectorizer = TfidfVectorizer( max_features = 40000, ngram_range = ( 1, 3 ), 
	sublinear_tf = True )

# vectorize train reviews
train_data_features = vectorizer.fit_transform( clean_train_reviews )

# vectorize test reviews
test_data_features = vectorizer.transform( clean_test_reviews )

# trains and calculate AUC 
def train_and_eval_auc( model, train_x, train_y, test_x, test_y ):
	model.fit( train_x, train_y )
	p = model.predict_proba( test_x )
	auc = AUC( test_y, p[:,1] )
	return auc

## testing penalization parameters
# for num in [1.0, 5.0, 10, 20]:
# 	lr = LR(penalty= "l2", C = num)
# 	auc = train_and_eval_auc( lr, train_data_features, train["sentiment"], \
# 		test_data_features, test["sentiment"].values )
# 	print("logit Reg AUC:", round(auc, 4), ", C = ", num)


# final model, 40K trigrams features, C = 5.0, l2 penalization.
# fit final model and make predictions
model = LR(C = 5.0)
model.fit(train_data_features, train['sentiment'])
preds = model.predict(test_data_features)

# create file for submission
output = pd.DataFrame( data = { 'id': test['id'], 'sentiment': preds } )
output.to_csv( output_file, index = False, quoting = 3 )
