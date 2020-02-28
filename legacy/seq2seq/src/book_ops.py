from __future__ import division
import sys, os, io
import re
import time
import string
import math
import numpy
import random
import json

import pickle 
from collections import defaultdict

# spaCy tools for parsing documents
import spacy
nlp = spacy.load('en')  

# root directories
base_dir = "../data/"
output_dir = base_dir + "interim/"

# source data
text_dir = base_dir + '/raw/amazon/'

from __future__ import division
import sys, os, io
import re
import time
import string
import math
import numpy
import random
import json

import pickle 
from collections import defaultdict

# spaCy tools for parsing documents
import spacy
nlp = spacy.load('en')  

# root directories
base_dir = "../data/"
output_dir = base_dir + "interim/"

# source data
text_dir = base_dir + '/raw/amazon/'

# processed paths
batch_path = base_dir + '/final/batches/'
train_data_path = base_dir + '/final/batch_train_data.pkl'
test_data_path = base_dir + '/final/batch_test_data.pkl'

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_UNK = b"_UNK"
_EOS = b"_EOS"
_EOD = b"_EOD"
_GO  = b"_GO"

_START_VOCAB = [_PAD, _UNK, _EOS, _EOD, _GO]

PAD_ID = 0
UNK_ID = 1
EOS_ID = 2
EOD_ID = 3
GO_ID = 4

def tokenize(doc, lower=False):
    return [w.text for w in nlp(doc)]

# iterate through the text, tokenize, get categories
# use one of the two flavors above
def read_data():
    descriptions = defaultdict(list)
    categories = defaultdict(list)

    cat_file = text_dir+'/info/categories.txt'
    with io.open(cat_file, 'r', encoding="ISO-8859-1") as c_file:
        print('  getting categories...')
        current = None
        for idx, line in enumerate(c_file.readlines()):
            if idx % 10000 == 0: print('\t%s' % idx)
            if line.startswith((' ', '\t')):
                tokens = [w.text.lower() for w in nlp(line.strip())]
                categories[current].append(tokens)
            else:
                current = line.strip()
            sys.stdout.flush()
        categories = {k: v for k,v in categories.items() if 
                      v.startswith('books')}
    with open(categories_path, 'wb') as c_out:
        Pickle.dump(categories, c_out, protocol=-1)

    desc_file = text_dir+'/info/descriptions.txt'
    with io.open(desc_file, 'r', encoding="ISO-8859-1") as d_file:
        print('\n  getting descriptions...')
        current = None
        texts, tokens, keys = [],[],[]
        for idx, line in enumerate(d_file.readlines()):
            items = line.split(':', 1)
            if items[0] == 'product/productId':
                keys.append(items[1].strip())
            if items[0] == 'product/description':
                texts.append(items[1])
        print('\n  tokenizing descriptions...')
        for doc in nlp.pipe(texts, n_threads=16, batch_size=10000):
            tokens.append(get_spacy_tokens(doc))
            if len(tokens) % 10000 == 0: print('\t%s' % len(tokens))
        descriptions = dict(zip(keys, tokens))
        with open(descriptions_small_path, 'wb') as d_out:
            Pickle.dump(descriptions, d_out, protocol=-1)
    return descriptions, categories
