from __future__ import division
import sys, os, io
import re
import time
import string
import math
import numpy
import random
import json
import itertools
import pickle as Pickle
from collections import defaultdict
from itertools import tee, zip_longest
from itertools import cycle

# NLTK tools for parsing documents
import nltk
from nltk import wordpunct_tokenize as tokenize
sdetect = nltk.data.load('tokenizers/punkt/english.pickle')
np_grammar = "NP: {<DT>?<JJ>*<NN>}"
np_parser = nltk.RegexpParser(np_grammar)

category_stop = ['books','new','a-z']

# spaCy tools for parsing documents
import spacy
nlp = spacy.load('en')  

# Pythonrouge for computing rouge scores
# from pythonrouge.pythonrouge import Pythonrouge

# root directories
base_dir = "/home/archive/ssharpe/development/hs2s/data/"
raw_dir = base_dir + "raw/"
interim_dir = base_dir + "interim/"
processed_dir = base_dir + "processed/"
train_batch_dir = processed_dir + "train_batches/"
test_batch_dir = processed_dir + "test_batches/"
dev_batch_dir = processed_dir + "dev_batches/"

# processed paths
data_path = interim_dir + 'pairs/'

train_data_path = processed_dir + 'train/'
test_data_path = processed_dir + 'test/'

# vocabulary path
encoder_vocab_path = processed_dir + 'vocab/encoder_vocab.txt'
decoder_vocab_path = processed_dir + 'vocab/decoder_vocab.txt'


# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_UNK = "_UNK"
_EOS = "_EOS"
_EOD = "_EOD"
_GO  = "_GO"

_START_VOCAB = [_PAD, _UNK, _EOS, _EOD, _GO]

PAD_ID = 0
UNK_ID = 1
EOS_ID = 2
EOD_ID = 3
GO_ID = 4


def load_data():
    encoder_tokens, decoder_tokens = [],[]    
    review_fn = processed_dir + 'book_review_data.json'
    with open(review_fn,'r') as review_file:
        reviews = json.load(review_file)
    return reviews
            
def build_vocabularies():
    reviews = load_data()
    encoder_tokens, decoder_tokens = [],[]
    for key, review in reviews.items():
        sents = [s.split() for s in review['sentences']]
        sents = [[x.strip() for x in s] for s in sents]
        cats = [s.split(',') for s in review['categories']]
        cats = [[x.strip() for x in s] for s in cats]
        for sent in sents:
            encoder_tokens.extend(sent)
        for cat in cats:
            decoder_tokens.extend([c.lower() for c in cat])
            
    encoder_vocab, decoder_vocab = {},{}
    def build_vocab(tokens):
        vocab = {}
        for token in tokens:
            token = token.strip()
            if token not in vocab:
                vocab[token] = 1
            else:
                vocab[token] += 1
        vocab_list = list(reversed(sorted(list(
            vocab.items()),key=lambda x:x[1])))
        return vocab_list
                            
    print('building encoder vocab...')
    encoder_vocab_list = build_vocab(encoder_tokens)
    print('encoder vocab: %s' % len(encoder_vocab_list))
    with open(encoder_vocab_path, 'w') as ev_file:  
      for w, count in encoder_vocab_list:
          ev_file.write('{0}\t{1}\n'.format(w, str(count)))

    print('building decoder vocab...')            
    decoder_vocab_list = build_vocab(decoder_tokens)
    print('decoder vocab: %s' % len(decoder_vocab_list))        
    with open(decoder_vocab_path, 'w') as dv_file: 
      for w, count in decoder_vocab_list:
          dv_file.write('{0}\t{1}\n'.format(w, str(count)))          

          
def show_stats():
    reviews = load_data()
    doc_lens, sen_lens, cat_nums, cat_depths = [],[],[],[]
    for key, review in reviews.items():
        doc_lens.append(len(review['sentences']))
        cat_nums.append(len(review['categories']))
        sents = [s.split() for s in review['sentences']]
        slens = [len(s) for s in sents]
        sen_lens.extend(slens)
        cats = [s.split(',') for s in review['categories']]
        cdepths = [len(s) for s in cats]
        cat_depths.extend(cdepths)
    avg_doc = sum(doc_lens)/len(doc_lens)
    max_doc = max(doc_lens)
    std_doc = numpy.std(numpy.array(doc_lens))
    print('Avg sens per doc: {}'.format(avg_doc))
    print('Max sens per doc: {}'.format(max_doc))
    print('Std sens per doc: {}'.format(std_doc))    
    avg_cat = sum(cat_nums)/len(cat_nums)
    max_cat = max(cat_nums)
    std_cat = numpy.std(numpy.array(cat_nums))
    print('\nAvg num cats per doc: {}'.format(avg_cat))
    print('Max cats per doc: {}'.format(max_cat))
    print('Std cats per doc: {}'.format(std_cat))    
    avg_sen = sum(sen_lens)/len(sen_lens)
    max_sen = max(sen_lens)
    std_sen = numpy.std(numpy.array(sen_lens))
    print('\nAvg tokens per sen: {}'.format(avg_sen))
    print('Max tokens per sen: {}'.format(max_sen))
    print('Std tokens per sen: {}'.format(std_sen))    
    avg_depth = sum(cat_depths)/len(cat_depths)
    max_depth = max(cat_depths)
    std_depth = numpy.std(numpy.array(cat_depths))
    print('\nAvg depth per cat: {}'.format(avg_depth))
    print('Max depth per cat: {}'.format(max_depth))
    print('Std depth per cat: {}'.format(std_depth))    
    

def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)


def divide_list(l, n):
    for i in range(0, len(l), n):
      yield l[i:i + n]


def load_vocabularies(encoder_size, decoder_size):
    encoder_vocab, decoder_vocab = {},{}
    def sort_vocab(lines,size):
      kv = [line.split('\t') for line in lines]
      kv_num = [(k,int(v.strip().rstrip())) for k,v in kv]
      kv_sort = list(reversed(sorted(kv_num, key=lambda x:x[1])))
      start = [(k, None) for k in _START_VOCAB]
      kv_ids = [(k,i) for i, (k,v) in enumerate(start + kv_sort)]
      if size:
          return kv_sort[:size], kv_ids[:size]
      else:
          return kv_sort, kv_ids
    print('loading vocabularies...')
    with open(encoder_vocab_path, 'r') as ev_file:
        encoder_lines = ev_file.readlines()
        encoder_counts, encoder_ids = sort_vocab(encoder_lines, encoder_size)            
    with open(decoder_vocab_path, 'r') as dv_file:
        decoder_lines = dv_file.readlines()
        decoder_counts, decoder_ids = sort_vocab(decoder_lines, None) 
    return encoder_counts, encoder_ids, decoder_counts, decoder_ids
      
      
def make_batches(vocabularies, batch_size, doc_size, split_size):
    print('Building training examples...')
    reviews = list(load_data().items())
    random.shuffle(reviews)
    batches = []
    encoder_counts, encoder_ids, decoder_counts, decoder_ids = vocabularies
    corpus_word_count = sum([x[1] for x in encoder_counts])
    encoder_probs = [(w, c/corpus_word_count) for w,c in encoder_counts]
    decoder_probs = [(w, c/corpus_word_count) for w,c in decoder_counts]    
    encoder_vocab_rev = {v:k for k,v in encoder_ids}
    decoder_vocab_rev = {v:k for k,v in decoder_ids}
    encoder_ids, decoder_ids = dict(encoder_ids), dict(decoder_ids)
    encoder_counts, decoder_counts = dict(encoder_counts), dict(decoder_counts)
    encoder_probs, decoder_probs = dict(encoder_probs), dict(decoder_probs)
    doc_count = 0
    encoder_inputs, decoder_inputs = [],[]
    t0 = time.time()
    for ridx, (key, review) in enumerate(reviews):
        if ridx and ridx % 10000 == 0:
            t1 = time.time()
            print('batches complete: {0} ({1})'.format(ridx, t1-t0))
        if ridx and ridx % 400000 == 0: break
        # get category ids
        cats = [s.split(',') for s in review['categories']]
        cat_ids, cats_filtered = [], []
        for cat in cats:
            cat_filt = [x.strip() for x in cat if
                        not any(t in x.lower() for t in category_stop)]
            if cat_filt:
                cats_filtered.append(cat_filt)
        if cats_filtered:
            for cat in cats_filtered:
                cat_id = []
                for subcat in cat:
                    w = subcat.lower().strip()
                    wid = decoder_ids[w]
                    wcount = decoder_counts[w]
                    wprob = decoder_probs[w]
                    wquad = (w, wid, wcount, wprob)
                    cat_id.append(wquad)
                cat_ids.append(cat_id)
            # get sentence ids        
            sents = [s.split() for s in review['sentences']]
            for doc_sents in window(sents, doc_size):
                sents_ids, doc_words = [],[]
                for sent in doc_sents:
                    sent_quad = [(w, encoder_ids[w], encoder_counts[w], encoder_probs[w])
                                    if w in encoder_ids else
                                    (_UNK, UNK_ID, None, None) for w in sent]
                    doc_words.extend([x[0] for x in sent_quad])
                    sent_quad += [(_EOS, EOS_ID, None, None)]
                    sents_ids.append(sent_quad)
                # ignore examples with > 4% unknown words
                unk_vec = [1 if w==_UNK else 0 for w in doc_words]
                unk_perc = sum(unk_vec)/len(unk_vec)
                if unk_perc > 0.04: continue
                if max(len(s) for s in sents_ids) > 100: continue
                sents_ids[-1].append((_EOD, EOD_ID, None, None))
                encoder_inputs.append(sents_ids)
                decoder_inputs.append(cat_ids)
    examples = list(zip(encoder_inputs, decoder_inputs))
    random.shuffle(examples)
    set_split = int(round(len(examples) * split_size))
    dev_set = examples[:set_split]
    train_set = examples[set_split:]
    train_batches, dev_batches = [],[]
    encoder_shape, decoder_shape = [],[]     
    print('total train examples: {}'.format(len(train_set)))
    for idx, batch in enumerate(divide_list(train_set, batch_size)):
        if idx % 1000 == 0: print(idx)
        num_word = max([max([len(x) for x in doc[0]]) for doc in batch])
        num_sen = max([len(doc[0]) for doc in batch]) 
        num_cat = max([max([len(x) for x in doc[1]]) for doc in batch])
        num_label = max([len(doc[1]) for doc in batch])         
        bfname = train_batch_dir + 'batch_{0}_{1}_{2}_{3}.json'.format(
            idx, num_sen, num_word, num_label, num_cat)
        if len(batch) == batch_size:
            with open(bfname,'w') as batch_file:
                json.dump(batch, batch_file)
    print('total dev examples: {}'.format(len(dev_set)))
    for idx, batch in enumerate(divide_list(dev_set, batch_size)):
        if idx % 1000 == 0: print(idx)
        num_word = max([max([len(x) for x in doc[0]]) for doc in batch])
        num_sen = max([len(doc[0]) for doc in batch]) 
        num_cat = max([max([len(x) for x in doc[1]]) for doc in batch])
        num_label = max([len(doc[1]) for doc in batch])         
        bfname = train_batch_dir + 'batch_{0}_{1}_{2}_{3}.json'.format(
            idx, num_sen, num_word, num_label, num_cat)
        if len(batch) == batch_size:
            with open(bfname,'w') as batch_file:
                json.dump(batch, batch_file)
                
            
def data_shape(source='train'):
    # batch_fns = os.listdir(train_batch_dir)
    batch_fns = os.listdir(test_batch_dir)    
    for batch_fn in sorted(batch_fns):
        shape = batch_fn.split('.')[0].split('_')[-2:]
        shape = [int(x) for x in shape]
        yield shape
                

def blocks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def train_iter(batch_size, shape=False):
    batch_fns = os.listdir(train_batch_dir)
    for batch_fn in sorted(batch_fns):
        with open('../data/processed/train_batches/'+batch_fn, 'r') as batch_file:
            batch_orig = json.load(batch_file)
            batches = blocks(batch_orig, batch_size)
            if not shape:
                for batch in batches:
                    yield batch
            else:
                examples = len(batch)
                emaxes = [max([len(s) for s in e]) for e,_ in batch]
                emaxsen = max([len(e) for e,_ in batch])
                emaxword = max(emaxes)
                dmaxes = [max([len(c) for c in d]) for _,d in batch]
                dmaxsen = max([len(d) for _,d in batch])
                dmaxword = max(dmaxes)
                yield examples, emaxsen, emaxword, dmaxsen, dmaxword


def dev_iter(batch_size, shape=False):
    batch_fns = os.listdir(dev_batch_dir)
    for batch_fn in sorted(batch_fns):
        with open('../data/processed/dev_batches/'+batch_fn, 'r') as batch_file:
            batch_orig = json.load(batch_file)
            batches = blocks(batch_orig, batch_size)
            if not shape:
                for batch in batches:
                    yield batch
            else:
                examples = len(batch)
                emaxes = [max([len(s) for s in e]) for e,_ in batch]
                emaxsen = max([len(e) for e,_ in batch])
                emaxword = max(emaxes)
                dmaxes = [max([len(c) for c in d]) for _,d in batch]
                dmaxsen = max([len(d) for _,d in batch])
                dmaxword = max(dmaxes)
                yield examples, emaxsen, emaxword, dmaxsen, dmaxword

                                  
        
def entropy(batch):
    entropies, counts = {},{}
    # iterate through batch once to compute probs
    
    probs = {w: float(c) / len(counts) for w, c in counts.items()}
    probs = {w: v for w, v in probs.items() if v > 0.}
    # iterate again to compute entropy for each sequence
    for idx, seq in enumerate(batch_input):
        ent = 0
        for p in [probs[w] for w in seq]:
            if p > 0:
                ent -= p * math.log(p, 2.)
        ents[idx] = ent
    return ents


def get_textdata(data):
  text_data = []
  first = True
  for word, attn in data:
      if word == '_EOD': break
      elif word == '_EOS': text_data.append(('.', numpy.array([0])))
      elif word in _START_VOCAB: continue
      else: text_data.append((word, attn))
  text = ' '.join([x[0] for x in text_data]).rstrip()
  text = text.replace(' .','.')
  return text, text_data


def get_text(data):
  text_data = []
  first = True
  for word in data:
      if word == '_EOD': break
      elif word == '_EOS': text_data.append('.')
      elif word in _START_VOCAB: continue
      else: text_data.append(word)
  text = ' '.join(text_data).rstrip()
  text = text.replace(' .','.')
  return text

            
def load_batches(batch_size, gen_cycle=True, shape=False):
  print('\nloading data...')
  sys.stdout.flush()
  train_batchset, dev_batchset = [],[]
  if gen_cycle:
    train_batches = cycle(train_iter(batch_size, shape))
    dev_batches = cycle(dev_iter(batch_size, shape))
    return train_batches, dev_batches
  else:
    return train_iter(shape), dev_iter(shape)
    

if __name__ == "__main__": 
    inpt =  {'sen': 16, 'word': 25} # text 15 sen, 24 words per sen  
    otpt =  {'sen': 2, 'word': 21} # category 1 sen, 20 words per sen
    model_size = {'in': inpt, 'out': otpt}
    
    # Read in tokens and make batches from token ids
    if sys.argv[1] == '-build_vocab':
        build_vocabularies()
        
    elif sys.argv[1] == '-compute_stats':
        show_stats()
        
    elif sys.argv[1] == '-make_batches':
        encoder_vocab_size = 150000
        decoder_vocab_size = 3878
        batch_size = 128
        doc_size = 10
        split_size = 0.10
        vocabularies = load_vocabularies(encoder_vocab_size, 
                                         decoder_vocab_size)
        # will only build examples if data doesn't exist 
        make_batches(vocabularies, batch_size, doc_size, split_size)

    elif sys.argv[1] == '-iter_batches':
        data_shape()
        
    elif '-build_json':
        batch_dictionaries()        
            
    elif sys.argv[1] == '-summary_scores':
        summarization_scores(summary_path=None)

    elif sys.argv[1] == '-make_tokens': tokenize()

    
