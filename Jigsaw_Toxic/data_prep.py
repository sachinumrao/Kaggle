import pathlib
import numpy as np
import pandas as pd
import contractions
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

stops = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
WORD = re.compile(r'\w+')


def wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)
    
    
def fix_contractions(s):
    s = contractions.fix(s)
    return s


def clean_text(s):
    # remove newline chars \n
    s = s.replace('\n', ' ')
    
    # remove punctuations
    s = re.sub(r'[^\w\s]', '', s) 
    
    # remove numbers
    s = s.translate({ord(ch): None for ch in '0123456789'})
    
    return s
 
    
def convert2lower(s):
    return s.lower()

   
def tokenize(s):
    tokens = WORD.findall(s)
    return tokens


def lemmatize_tokens(s):
    s = [lemmatizer.lemmatize(w, wordnet_pos(w)) for w in s]
    return s


def filter_stopwords(s):
    s = [word for word in s if word not in stops]
    return s


def clean_data(df):
    # fix contractions
    df['clean_text'] = df['comment_text'].apply(lambda x: fix_contractions(x))
    
    # clean the text
    df['clean_text'] = df['clean_text'].apply(lambda x: clean_text(x))
    
    # convert text to lowercase
    df['clean_text']  = df['clean_text'].apply(lambda x: convert2lower(x))
    
    # tokenize text
    df['clean_text'] = df['clean_text'].apply(lambda x: tokenize(x))
    
    # lemmatize text
    df['clean_text'] = df['clean_text'].apply(lambda x: lemmatize_tokens(x))
    
    # filter stop words
    df['clean_text'] = df['clean_text'].apply(lambda x: filter_stopwords(x))
    
    return df


if __name__ == "__main__":
    fname = pathlib.Path.home().joinpath('Data', 'toxic', 'train.csv')
    train = pd.read_csv(fname, nrows=100)
    train = clean_data(train)
    train.to_csv('cleant_train.csv', index=False)