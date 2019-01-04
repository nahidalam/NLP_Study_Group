# https://nlpforhackers.io/training-pos-tagger/
# supervised approach for building a POS tagger

# unsupervised approach  using HMM
# https://ou.monmouthcollege.edu/_resources/pdf/academics/mjur/2014/Unsupervised-Part-of-Speech-Tagging.pdf
# semi-supervised approach https://arxiv.org/ftp/arxiv/papers/1407/1407.2989.pdf

'''
The intuition behind all stochastic taggers is simple generalization of the “pick the most-likely tag
for this word”. For a given sentence or a word sequence, HMM tagger chooses the tag sequence
that maximizes:
P (word | tag) * P (tag | previous n tags).

check the architecture at p - 17 https://arxiv.org/ftp/arxiv/papers/1407/1407.2989.pdf
'''

import nltk
import pprint
from sklearn.tree import DecisionTreeClassifier

# we are taking some tagged sentenses from nltk

tagged_sentences = nltk.corpus.treebank.tagged_sents()

'''

Before starting training a classifier, we must agree first on what features to
use. Most obvious choices are: the word itself, the word before and the word
after. That’s a good start, but we can do so much better. For example,
the 2-letter suffix is a great indicator of past-tense verbs, ending in “-ed”.
3-letter suffix helps recognize the present participle ending in “-ing”.

'''

def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

# split the dataset into train, test set
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

# do the necessary fit and transform
'''
 - create a vectorizer object tfidf or count
 - vectorizer.fit_transform(train)
 - vectorizer.transform(test)
'''
# seperate X, y

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)
print ("Accuracy:", clf.score(X_test, y_test))

# at this point, model is done

# lets test our model by tagging a sentence

def pos_tag(sentence):
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return zip(sentence, tags)

print pos_tag(word_tokenize('This is my friend, John.'))
# [('This', u'DT'), ('is', u'VBZ'), ('my', u'JJ'), ('friend', u'NN'), (',', u','), ('John', u'NNP'), ('.', u'.')]


'''
pprint.pprint(features(['This', 'is', 'a', 'sentence'], 2))

{'capitals_inside': False,
 'has_hyphen': False,
 'is_all_caps': False,
 'is_all_lower': True,
 'is_capitalized': False,
 'is_first': False,
 'is_last': False,
 'is_numeric': False,
 'next_word': 'sentence',
 'prefix-1': 'a',
 'prefix-2': 'a',
 'prefix-3': 'a',
 'prev_word': 'is',
 'suffix-1': 'a',
 'suffix-2': 'a',
 'suffix-3': 'a',
 'word': 'a'}
'''
