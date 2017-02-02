# to run in the command line
# first argument file name, second argument the output file name

import pandas as pd
from bs4 import BeautifulSoup
import sys

# read text document to pandas data frame.
review_data = pd.read_csv(sys.argv[1], header=0, delimiter='\t', quoting=3)

# function to remove html tags from raw data.  
def remove_html(raw_review):
    return(BeautifulSoup(raw_review, 'lxml').get_text())

# to append with text.  
cleaned_text = []

print('Parsing review data..')

# remove HTML from each movie review.  
for i in range(0, len(review_data)):
    cleaned_text.append(remove_html(review_data['review'][i]))

# remove orginal text.  
review_data = review_data.drop('review', 1)

# add parsed texted to train data.  
review_data['review_parsed'] = cleaned_text

# output file path.  
output_file_path = './data/'+sys.argv[2]+'.csv'

# export csv file.   
review_data.to_csv(output_file_path, index = False)

print('Copied file to', output_file_path)
