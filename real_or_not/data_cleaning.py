import numpy as np
import pandas as pd

import contractions
import string
import re
import spacy

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

from collections import Counter
from unidecode import unidecode


# Define utility class for text cleaning
class DataCleaner:

    def __init__(self):
        self.stops = self.get_stopwords()
        self.tokenizer = RegexpTokenizer(r"\w+")
        self.wordnet_lemmatizer = WordNetLemmatizer()

    def get_stopwords(self):
        nlp = spacy.load('en')
        stops = nlp.Defaults.stop_words
        retain_words = ['always', 'nobody', 'cannot', 'none', 'never', 'no', 'not']
        for j in retain_words:
            stops.discard(j)
        return stops

    def remove_non_ascii(self, text):
        ascii_text = text.encode('ascii', 'ignore').decode()
        return ascii_text

    def get_wordnet_pos(selfself, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    def cleaner(self, text):
        # Remove html links
        text = re.sub(r"http\S+", '', text, flags=re.MULTILINE)
        # Remove non-ascii characters
        text = self.remove_non_ascii(text)
        # Fix contractions and convert to lowercase
        text = contractions.fix(text).lower()
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        # Remove stopwords
        # tokens = [tok for tok in tokens if tok not in self.stops]
        # Lemmatize tokens
        tokens = [self.wordnet_lemmatizer.lemmatize(tok, self.get_wordnet_pos(tok)) for tok in tokens]
        # Remove numbers from tokens
        tokens = [tok for tok in tokens if tok.isalpha()]
        # Remove unit length tokens
        tokens = [tok for tok in tokens if len(tok) > 1 and tok != 'i']
        # Rejoin tokens to form string
        cleaned_text = " ".join(tokens)

        return cleaned_text


# Main function
def main(input_fname, output_fname):
    df = pd.read_csv(input_fname)

    # Drop columns
    df = df.drop(['id', 'keyword', 'location'], axis=1)

    # Create data cleaner
    cleaner = DataCleaner()

    # Apply cleaner on text column
    df['clean_text'] = df['text'].apply(lambda x: cleaner.cleaner(x))

    # Save cleaned text file
    df.to_csv(output_fname, index=False)


if __name__ == "__main__":
    # Define input and output files
    # Training data
    input_file = "~/Data/Kaggle/real_or_not/train.csv"
    output_file = "~/Data/Kaggle/real_or_not/clean_train.csv"
    main(input_file, output_file)
    print("Finished Cleaning Training Data...")

    # Testing data
    test_input_file = "~/Data/Kaggle/real_or_not/test.csv"
    test_output_file = "~/Data/Kaggle/real_or_not/clean_test.csv"
    main(test_input_file, test_output_file)
    print("Finished Cleaning Testing Data...")


