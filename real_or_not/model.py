from collections import Counter
from nltk.tokenize import RegexpTokenizer
from unidecode import unidecode
import numpy as np
import pandas as pd
import contractions
import string
import re

train_file = '~/Data/Kaggle/real_or_not/train.csv'
df = pd.read_csv(train_file)

df = df.drop(['id', 'keyword', 'location'], axis=1)

tokenizer = RegexpTokenizer(r"\w+")

## Utility functions
def remove_non_ascii(text):
    return text.encode("ascii", "ignore").decode()

def clean_text(text):
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = remove_non_ascii(text)
    text = contractions.fix(text).lower()
    tokens = tokenizer.tokenize(text)
    return tokens


## 
df['tokens'] = df['text'].apply(lambda x: clean_text(x))

token_farm = df['tokens'].to_list()

flat_list = [token for sublist in token_farm for token in sublist]

token_cnt = Counter(flat_list)
