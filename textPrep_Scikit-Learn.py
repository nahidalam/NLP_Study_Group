'''
Word Counts with CountVectorizer - only considers count, throws away the word order
Not interesting
'''


'''
Word Frequencies with TfidfVectorizer

Reference: http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
'''

from sklearn.feature_extraction.text import TfidfVectorizer

#text = "This is a virtual NLP study group. We love NLP. We will do interesting projects"
text = ["This is a virtual NLP study group.","We love NLP.","We will do interesting projects"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
#a vocabulary of 12 words is learned from the documents
#each word is assigned a unique integer index in the output vector
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
#transforms to a tf-idf representation.
#vector = vectorizer.transform(text[0])
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
#print(vector.toarray())
