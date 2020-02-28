"""
Text Processing Pipeline
"""

import os, sys
import re
import json
import pickle
import numpy as np
import pprint
import argparse
import time
import hashlib 
import string
from datetime import datetime
from joblib import Parallel, delayed
from functools import partial
from collections import Counter

# initialize language model
# https://spacy.io/models/en#en_vectors_web_lg
import spacy
from spacy.util import minibatch
nlp = spacy.load('en_core_web_sm')
table = str.maketrans({key: ' ' for key in string.punctuation})
# nltk stopwords & collocation

import nltk
from nltk.collocations import *
from nltk.corpus import stopwords
stop = stopwords.words('english')

from html.parser import HTMLParser

def mkdir(dir_path):
    """ make a directory if not exists """
    if not os.path.exists(dir_path):
        return os.makedirs(dir_path)
                        
class MLStripper(HTMLParser):
    """ strip meta tags from HTML string """
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

    
def strip_tags(html):
    """ feeder for HTML metatag stripper """
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def doc_load(doc_paths):
    """ Load JSON from file """
    docs = []
    for doc_path in doc_paths:
        with open(doc_path, 'r') as f:
            d = json.load(f)
            docs.extend(d)
    return docs

    
def doc_iter(doc_paths):
    """ Document iterator 
    """
    docs = doc_load(doc_paths)
    for d in docs:
        # print("\n-------------------- DOC ---------------------")
        text = d['text']
        yield text, d

        
def feedback_load(feedback_path):
    """ Load feedback 
    Not Implemented
    """
    with open(feedback_path, 'r') as f:
        feedback = json.load(f)
    return feedback


def feedback_iter(feedback_path):
    """ Iterator for feedback 
    Not Implemented
    """
    docs = feedback_load(feedback_path)
    for i, (key, feedback) in enumerate(docs.items()):
        yield feedback, key

                
def is_sent_begin(word):
    """ detection for sentence beginning
    Args: Doc word
    Return: BOOL is_beginning
    """
    if word.i == 0:
        return True
    elif word.i >= 2 and word.nbor(-1).text in (".", "!", "?", "..."):
        return True
    else:
        return False

    
def represent_word(word, ents, lemma=False, true_case=False):
    """ customized word representation 
    Args: 
      word: Spacy Token
    Returns: 
      string (other) ngram to word 
    """
    if str(word.ent_type_) in ents:
        text = word.ent_type_
    else:
        text = word.text.translate(table) if not lemma else word.lemma_.translate(table)
        text = text if not any(c.isdigit() for c in text) else 'NUM'
        # remove punctuation and sub-word stop
        text = ' '.join(cs for cs in text.split() if len(cs)>1 and cs not in stop)
        if true_case:
            # True-case, i.e. try to normalize sentence-initial capitals.
            if (
                    text.istitle()
                    and is_sent_begin(word)
                    # Only do this if the lower-cased form is more probable.
                    # and word.prob < word.doc.vocab[text.lower()].prob
            ): 
                text = text.lower()
            # Convert upper case strings to title case
            if text.isupper():
                text = text.title()
    return text


def transform_texts(nlp, batch_id, texts, batch_loc,
                    batch_size=10000, n_threads=11, print_data=False):    
    """ text transformation pipeline with spaCy multithreaded nlp model
    Args: nlp: loaded language model
    batch_id: the document batchID is passed in on execution
    : to enable multi-threading this function is called in delayed execution
    texts: iterator yielding (text to be tokenized, associated metadata)
    Returns: JSON doc object written out
    """
    ents = ['DATE','TIME','MONEY','QUANTITY','CARDINAL','EVENT','GPE','NORP','LOC',
            'PERSON', 'LANGUAGE','LAW','PERCENT','ORDINAL']
    print("Processing batch: {}".format(batch_id), end='\r')
    batch_path = batch_loc+"batch_{}.jsonl".format(batch_id)
    with open(batch_path, 'w') as f:                
        # tokenize using spaCy's built-in pipe() method
        # the description is tokenized in the pipe, everything else comes in metadata
        for doc, d in nlp.pipe(texts, n_threads, batch_size):
            try:
                # document tokens per sentence
                d['tokens'] = [[t.text for t in s] for s in doc.sents]
                # bag-of-words representation of document
                bow = [represent_word(t, ents) for t in doc]
                # lower case words greater than 2 chars that are not named entities (uppercase)
                d['bow'] = [t.lower() for t in bow if len(t) > 2 and not t.isupper()]
                f.write('{}{}'.format(json.dumps(d),'\n'))
            except Exception as e:
                print("\nFailed to transform document text:\n{}\n".format(str(e)))
                raise

            
def tokenize(doc_iter, doc_paths, batch_loc, batch_size=500):
    """ spaCy nlp model to tokenize dataframe - parse, pos, ner, dep
    Args: dataframe (rows of documents (each col with small sentence-size text)
    Returns: document dictionary batch file: confg['BATCH']
    : the interim document (between parse and load): JSONL
    : parser is multithreaded: nlp.pipe()
    : https://spacy.io/usage/examples
    """
    docs = {}
    print("Tokenizing documents...")
    t_start = time.time()
    # TODO: batch-size setting seems auto-determined, minibatch size not actually being set.
    partitions = minibatch(list(doc_iter(doc_paths)), size=batch_size)
    executor = Parallel(n_jobs=10, backend="multiprocessing", prefer="processes")
    do = delayed(partial(transform_texts, nlp))
    tasks = (do(i, batch, batch_loc) for i, batch in enumerate(partitions))
    executor(tasks)
    t_end = time.time()
    print("Completed in {} seconds\n".format(t_end-t_start))

    
def load_batchdata(batch_loc):
    parsed_docs = []
    batch_files = [os.path.join(batch_loc, f) for f in os.listdir(batch_loc)]
    for i, batch_file in enumerate(batch_files):
        print("Loading batch {} of {}".format(i, len(batch_files)), end='\r')
        with open(batch_file, "r") as fh:
            batch_lines = [line.strip().rstrip() for line in fh.readlines()]
            batch = [json.loads(line) for line in batch_lines]
            parsed_docs.extend(batch)
    return parsed_docs

    
def build_vocab(docs):
    """ build vocabulary from documents """
    tokens = [d['bow'] for d in docs]
    tokens = [t.lower() for d in tokens for t in d]
    vocab_freq = Counter(tokens)
    # exclude vocabulary words that occur only once
    vocab_list = sorted(vocab_freq, key=vocab_freq.get, reverse=True)
    vocab = {w: i for i, w in enumerate(vocab_list)}
    print("Vocabulary size: {}".format(len(vocab)))
    return vocab, vocab_list


def load_vocab(vocab_path):
    with open(vocab_path, 'r') as fh:
        return json.load(fh)
    
        
def main(args):
    doc_paths = [os.path.join(args.doc_dir, f) for f in os.listdir(args.doc_dir)]
    batch_loc = args.doc_dir + '/batches/'
    mkdir(batch_loc)
    tokenize(doc_iter, doc_paths, batch_loc)
    docs = load_batchdata(batch_loc)
    vocab, _ = build_vocab(docs)
    with open(args.model_dir + '/parsed_documents.json', 'w') as fh:
        json.dump(docs, fh)
    with open(args.model_dir + '/vocab.json', 'w') as fh:
        json.dump(vocab, fh)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Natural Language Processing Pipeline.')    
    args = parser.parse_args()
    main(args)
