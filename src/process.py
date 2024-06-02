import nltk
import spacy
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_dictionary(dict_name, lang, tokenizer, stopword=False):
    try:
        nlp = spacy.load(dict_name)
    except:
        os.system('python -m spacy download '+dict_name)
        nlp = spacy.load(dict_name)
    if stopword:
        try:
            eng_stopwords = nltk.corpus.stopwords.words(lang)
        except:
            nltk.download('stopwords')
            eng_stopwords = nltk.corpus.stopwords.words(lang)
    else:
        eng_stopwords = None
    try:
        _ = nltk.tokenize.word_tokenize('test')
    except:
        nltk.download(tokenizer)
    return nlp, eng_stopwords

def preprocess_text_and_store(text, doc_store=None, store=False, path='data/', eng_stopwords=None, nlp=None, COLAB=False):
    preprocessed_text = np.empty(len(text), dtype=object)
    if doc_store is None or doc_store not in os.listdir(path):
        counter = 0
        for i in range(len(text)):
            process_words = []
            text[i] = text[i].replace('\d', ' ')
            for word in nltk.word_tokenize(nlp(text[i].lower()).text):
                if word.isalpha() and word not in eng_stopwords and len(str(word)) >= 3:
                    process_words.append(word)
            preprocessed_text[counter] = ' '.join(process_words)
            counter += 1
    else:
        preprocessed_text = np.array(pd.read_csv(path+doc_store)['comment_text'])
    if store and not COLAB:
        pd.DataFrame(data = {"comment_text":preprocessed_text}).to_csv(path+doc_store)

    return preprocessed_text

def get_sequences(train_data, val_data, test_data, num_words=1000, verbose=1):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train_data)
    if verbose:
        print(f"Words number: {tokenizer.word_counts}")
    train_data = tokenizer.texts_to_sequences(train_data)
    test_data  = tokenizer.texts_to_sequences(test_data)
    val_data   = tokenizer.texts_to_sequences(val_data)
    vocab_size = len(tokenizer.word_index) + 1
    if verbose:
        print(f"Vocabulary size: {vocab_size}")
    maxlen = len(max(train_data, key=len))
    train_data = pad_sequences(train_data, maxlen=maxlen)
    test_data = pad_sequences(test_data, maxlen=maxlen)
    val_data = pad_sequences(val_data, maxlen=maxlen)
    if verbose:
        print(f"Pre-padded sequences: {train_data[0:5]}")
    return train_data, val_data, test_data, vocab_size, maxlen      