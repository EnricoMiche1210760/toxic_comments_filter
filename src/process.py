import nltk
import spacy
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_dictionary(dict_name, lang, tokenizer, stopword=False):
    '''
    Function to load the dictionary and the tokenizer
    Parameters:
        dict_name: str, name of the dictionary to be loaded
        lang: str, language of the dictionary
        tokenizer: str, name of the tokenizer to be loaded
        stopword: bool, if True, stopwords will be loaded
    Returns:
        nlp: spacy.lang, dictionary loaded
        eng_stopwords: list, stopwords loaded
    '''
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

def preprocess_text_and_store(text, doc_store=None, store=False, path='data/', eng_stopwords=None, nlp=None, COLAB=False, word_min_len=3):
    '''
    Function to preprocess the text and store it in a csv file
    Parameters:
        text: list, list of strings to be preprocessed
        doc_store: str, name of the csv file to store the preprocessed text
        store: bool, if True, the preprocessed text will be stored in a csv file
        path: str, path to store the csv file
        eng_stopwords: list, list of stopwords to be removed
        nlp: spacy.lang, dictionary to be used for lemmatization
        COLAB: bool, set to True if running on Google Colab
        word_min_len: int, minimum length of the words to be kept
    Returns:
        preprocessed_text: np.array, preprocessed text
    '''
    preprocessed_text = np.empty(len(text), dtype=object)
    if doc_store is None or doc_store not in os.listdir(path):
        counter = 0
        for i in range(len(text)):
            process_words = []
            text[i] = text[i].replace('\d', ' ')
            for word in nltk.word_tokenize(nlp(text[i].lower()).lemma_):
                if word.isalpha() and word not in eng_stopwords and len(str(word)) >= word_min_len:
                    process_words.append(word)
            preprocessed_text[counter] = ' '.join(process_words)
            counter += 1
    else:
        preprocessed_text = np.array(pd.read_csv(path+doc_store)['comment_text'])
    if store and not COLAB:
        pd.DataFrame(data = {"comment_text":preprocessed_text}).to_csv(path+doc_store)

    return preprocessed_text

def get_sequences(train_data, val_data, test_data, num_words=1000, verbose=1):
    '''
    Function to get the sequences of the text data
    Parameters:
        train_data: list, list of strings to be used for training
        val_data: list, list of strings to be used for validation
        test_data: list, list of strings to be used for testing
        num_words: int, number of words to be used in the tokenizer
        verbose: int, if 1, print the number of words and the vocabulary size
    Returns:
        train_data: np.array, sequences of the training data
        val_data: np.array, sequences of the validation data
        test_data: np.array, sequences of the testing data
        vocab_size: int, vocabulary size
        maxlen: int, maximum length of the sequences
    '''
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