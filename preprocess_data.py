"""
File for preprocessing Russian text
"""

import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import nltk
# Removing stop words
# nltk.download("stopwords")  # run it once
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")
# Stemming and Lemmatization
from pymystem3 import Mystem
stem = Mystem()
from string import punctuation



#%% Function for preprocess text

def preprocess_text(text):

    tokens = stem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and token.strip() not in punctuation]
    
    text = " ".join(tokens)
    
    return text


#%% Data load

train_data_path = 'dataset/train.tsv'
test_data_path = 'dataset/test.tsv'

def import_data(train_path, test_path):

    separator = None
    data_extention = os.path.splitext(train_data_path)[-1]

    if data_extention == '.csv':
        separator = ','
    elif data_extention == '.tsv':
        separator = '\t'
    else:
        assert separator, 'Unknown file format! Available: .csv and .tsv'

    train_df = pd.read_csv(train_path, sep=separator)
    test_df = pd.read_csv(test_path, sep=separator)

    print('Train file size: %i' % train_df.size)

    return train_df, test_df

train_df, test_df = import_data(train_data_path, test_data_path)


#%% Preprocess and save data

def preprocess_data(train, test, save=True):

    # It takes time (a lot of)
    train.iloc[:, 0] = train.iloc[:, 0].progress_apply(preprocess_text)
    test.iloc[:, 0] = test.iloc[:, 0].progress_apply(preprocess_text)

    train.to_csv('dataset/train_preprocessed.csv', index=False)
    test.to_csv('dataset/test_preprocessed.csv', index=False)

    return train, test

train, test = preprocess_data(train_df, test_df, save=True)


"""
Possible additional options:
# add Named-entity recognition (https://deeppavlov.ai/)

# Another tools
# import pymorphy2
# from nltk.tokenize import word_tokenize
"""