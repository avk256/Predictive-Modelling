import sys
### change to the current script folder
sys.path.insert(0, 'E:\\Work\\aspirantura\\kafedra\\upwork\\RtoP\\PyModels\\LinearRegression')

from R2P_linregr_dataprep import columns_data_type, ch_sq_test, correlations, transformation, variable_importance_h2o, transformation_inv 

from nltk.corpus import TwitterCorpusReader
import logging
import gensim
import pickle  # noqa: E402
from gensim.test.utils import datapath
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import nltk; nltk.download('stopwords')
import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from nltk.corpus import stopwords
from gensim import models, similarities
import json



stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

###############################################################################
"""
Text tokenization method
input:
    sentences - text column in the dataset
output:
    list of the tokens
"""
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

"""
removeing stop words method
input:
    texts - list of tokens
output:
    list of the tokens without stopwords
"""
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

#def make_bigrams(texts,bigram_mod):
#    return [bigram_mod[doc] for doc in texts]
#
#def make_trigrams(texts,bigram_mod,trigram_mod):
#    return [trigram_mod[bigram_mod[doc]] for doc in texts]

"""
method for obteining normal forms of the words 
input:
    texts - list of tokens 
output:
    list of the tokens in normal word form
"""
def lemmatization(texts, nlp ,allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def tuple_to_vector(tuple_list, num_topics):
    topic_vector = [0] * (num_topics)
    for t in tuple_list:
        topic_vector[t[0]] = t[1]
    return topic_vector
    
"""
Text preprocessing method
input:
    data - text column in the dataset
output:
    corpus_bow - text after preprocessing in BOW format
    corpus_tfidf - - text after preprocessing in TFIDF format
    id2word - dictionary
"""
def clean_text(data):
    data_words = list(sent_to_words(data))
    data_words_nostops = remove_stopwords(data_words)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    data_lemmatized = lemmatization(data_words_nostops, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    data_lemmatized = [text for text in data_lemmatized]
    id2word = corpora.Dictionary(data_lemmatized)
    
    corpus_bow = [id2word.doc2bow(text) for text in data_lemmatized]
    tfidf = models.TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]
    
    return corpus_bow, corpus_tfidf, id2word 


"""
Text analysis method
input:
    corpus - text after preprocessing in BOW or TFIDF format
    id2word - dictionary
    num_topics - suggested number of topics
    num_topics - number of key words in each topic
output:
    terms in each topic
    data with topics

"""
def lda_model(corpus, id2word, num_topics, num_term):
    
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word = id2word, passes=20, num_topics=num_topics)
    ### the key words for each topic
    topics = lda.print_topics(num_topics, num_words=num_term)
    
    key_words = [0] * len(topics)
    
    for j in range(0,len(topics)):
        words = []
        for i in (topics[j][1].split('+')):  
            words.append(i.split('*')[1])
        key_words[j] = words
    ### get topic distribution for each document
    doc_topic = []
    for nn in range(0, len(corpus)):
    #    print('id',nn)
    #    print('text\n',data_lemmatized[nn])
        tmp = lda.get_document_topics(corpus[nn])
    #    print('topics\n',doc_topic)
        doc_topic.append(tuple_to_vector(tmp,num_topics))
    return key_words, doc_topic

"""
Text analysis method
input:
    corpus - text after preprocessing in BOW or TFIDF format
    id2word - dictionary
    num_topics - suggested number of topics
    num_topics - number of key words in each topic
output:
    terms in each topic
    data with topics

"""
def sentiment_analysis(data_text, data, doc_topic, sent_col):
    xTrain, xTest, yTrain, yTest = train_test_split(doc_topic, data_text[sent_col].iloc[data.index], test_size=0.4, random_state=42)
    print(np.shape(xTrain))
    print(np.shape(xTest))
    print(np.shape(yTrain))
    print(np.shape(yTest))
    
    clf = RandomForestClassifier(n_estimators=300, max_depth=10,random_state=0)
    clf.fit(xTrain, yTrain)
    score = clf.score(xTest, yTest, sample_weight=None)
    print('score', score)
    f1 = f1_score(yTest, clf.predict(xTest), average='weighted')
    print('f1_score', f1)
    return score, f1
    
    

###############################################################################
    


###############################################################################

"""
Text analysis method
input:
    #* @param dbHost 
    #* @param dbPort 
    #* @param userName 
    #* @param password
    #* @param dbName 
    #* @param query
    #* @param textColumns
    #* @param parametersObj
    #* @param columnsArray
    #* @get /text_analysis
output:    
"""
def text_analysis(dbHost="",dbPort="",userName="",password="",dbName="",query="",textColumns="",parametersObj="",columnsArray=""):
    ##############
    # connecting to BD
    ##############

    data_text = pd.read_csv("E:/Work/aspirantura/kafedra/upwork/RtoP/Models_sep/Datasets/googleplaystore_user_reviews.csv")
    data, columnsArray_e = columns_data_type(data_text, columnsArray)

    data = data_text[textColumns].dropna()

    corpus_bow, corpus_tfidf, id2word = clean_text(data)
    key_words, doc_topic = lda_model(corpus=corpus_bow, id2word=id2word, num_topics=5, num_term=10)
    sentiment = sentiment_analysis(data_text=data_text, data=data, doc_topic=doc_topic, sent_col=parametersObj)
    
    return [sentiment, doc_topic, key_words]
    

####################### testing text analysis

columnsArray = '[{"columnDisplayName":"App","tableDisplayType":"string","columnName":"App"},{"columnDisplayName":"Translated_Review","tableDisplayType":"string","columnName":"Translated_Review"},{"columnDisplayName":"Sentiment","tableDisplayType":"string","columnName":"Sentiment"},{"columnDisplayName":"Sentiment_Polarity","tableDisplayType":"number","columnName":"Sentiment_Polarity"},{"columnDisplayName":"Sentiment_Subjectivity","tableDisplayType":"number","columnName":"Sentiment_Subjectivity"}]'    



try:
    columnsArray = json.loads(columnsArray)
except:
    print('json format not valid')



text_model = text_analysis(dbHost="",dbPort="",userName="",password="",dbName="",query="",textColumns="TranslatedReview",parametersObj="Sentiment",columnsArray=columnsArray)



