'''
Author: Nahid Alam

This document contains various examples of text cleaning using NLTK library

Installation:
1. Make sure you have Python installed. This code is tested on Python3
2. To install NLTK: sudo pip install -U nltk
3. Install data used with the NLTK library. Put below line in the script
import nltk
nltk.download()

OR use below in the command line
python -m nltk.downloader all

Reference:

1. Deep Learning for NLP
2. Processing Raw Text, Natural Language Processing with Python.
http://www.nltk.org/book/ch03.html
'''


#split a text into words

from nltk.tokenize import word_tokenize

text = "This is a virtual NLP study group. We love NLP. We will do interesting projects. bob's"
tokens = word_tokenize(text)
#print(tokens)

# remove all tokens that are not alphabetic
words = [word for word in tokens if word.isalpha()]
#print(words)


#filter out stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
#print(stop_words)

#stem words
#stemming is finding the root, helpful for sentiment analysis
#Porter Stemming algorithm is one of the stemming algorithms
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in tokens]
print(stemmed)
