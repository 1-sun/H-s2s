from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import copy
import string
import json
import pickle as Pickle


from termcolor import colored
from collections import Counter
from  more_itertools import unique_everseen as uniq

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


import data_ops
import hs2s_model

tf.app.flags.DEFINE_float("split", 0.1, "train/dev split")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "learning rate decay.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size to use during training.")
tf.app.flags.DEFINE_integer("lstm_size", 1000, "size of each model layer.")
tf.app.flags.DEFINE_float("dropout", 0.2, "dropout keep prob (1 is no dropout).")
tf.app.flags.DEFINE_integer("num_layers", 2,  "number of layers in the rnn cell.")
tf.app.flags.DEFINE_integer("encoder_vocab_size", 120000, "encoder vocabulary size.")
tf.app.flags.DEFINE_integer("decoder_vocab_size", 7000, "decoder vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "training directory.")
tf.app.flags.DEFINE_string("name", "hs2s", "name for the model when saving.")
tf.app.flags.DEFINE_integer("max_data_size", 0, "limit on size of data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100, "training steps per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False, "set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False, "run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("reverse_input", False, "reverse sequence inputs if True.")
tf.app.flags.DEFINE_boolean("compute_entropy", False, "compute sequence entropy from a random batch")

FLAGS = tf.app.flags.FLAGS

inpt =  {'sen': 10, 'word': 40} # text 10 sen, 40 words per sen  
otpt =  {'sen': 5, 'word': 8} # category 5 sen, 8 words per sen

model_size = {'in': inpt, 'out': otpt}

processed_data_path = FLAGS.data_dir + '/processed/book_review_data.json'
model_data_path = FLAGS.data_dir+FLAGS.name
model_train_data_path = model_data_path + '/train_data.json'
model_dev_data_path = model_data_path + '/dev_data.json'
checkpoint_data_path = model_data_path + '/checkpoint_data/'

def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  print("Creating model...")
  model = hs2s_model.HS2S(FLAGS.encoder_vocab_size, FLAGS.decoder_vocab_size, 
                          model_size, FLAGS.lstm_size, FLAGS.num_layers, FLAGS.dropout,
                          FLAGS.max_gradient_norm, FLAGS.batch_size,
                          FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
                          forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path+'.meta'):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model

    
def train():

  print("\nmodel name: ", FLAGS.name)
  print("data directory: ", FLAGS.data_dir)
  print("model directory: ", FLAGS.train_dir)
  print("batch size: ", FLAGS.batch_size)
  print("encoder vocab size: ",FLAGS.encoder_vocab_size)
  print("decoder vocab size: ",FLAGS.decoder_vocab_size)
  print("hidden layer size: ", FLAGS.lstm_size)
  print("hidden layers per cell: ", FLAGS.num_layers)
  print("dropout keep prob: ", FLAGS.dropout)
  print("maximum gradient norm: ", FLAGS.max_gradient_norm)
  print("learning rate: ", FLAGS.learning_rate)
  print("learning rate decay: ", FLAGS.learning_rate_decay_factor)
  print("reversed inputs: ", FLAGS.reverse_input)
  
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  train_batches, dev_batches = data_ops.load_batches(FLAGS.batch_size)
  encoder_counts, encoder_ids, decoder_counts, decoder_ids  = data_ops.load_vocabularies(
    FLAGS.encoder_vocab_size, FLAGS.decoder_vocab_size)
  
  with tf.Session(config=config) as sess:
    # Create model.
    print("\nCreating %d layers of %d units..." % (FLAGS.num_layers*4, FLAGS.lstm_size))
    sys.stdout.flush()
    model = create_model(sess, False)

    step_time, loss = 0.0, 0.0
    current_step = 0 #model.global_step.eval()
    previous_losses = []

    while True:
      # Get a batch and make a step.
      print('step: %s' % current_step, end='\r')
      t0 = time.time()
      sys.stdout.flush()
      start_time = time.time()
      input_data, target_data, target_weights = model.get_batch(next(train_batches), 
                                                                reverse_input=FLAGS.reverse_input)
      step_loss = model.step(sess, input_data, target_data, target_weights, 
                             forward_only=False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint


      # checkpoint and eval
      current_step += 1
      if current_step % FLAGS.steps_per_checkpoint == 0:
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("\nglobal step %d learning rate %.4f step-time %.2f" 
               % (model.global_step.eval(), model.learning_rate.eval(), step_time))
        print ("  train perplexity %.2f" % perplexity)
        print ("  train cost %.10f" % loss)
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.name+".ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        input_data, target_data, target_weights = model.get_batch(next(dev_batches), 
                                                                  reverse_input=FLAGS.reverse_input)
        eval_loss, attns, output_logits = model.step(sess, input_data, target_data, 
                                                     target_weights, 
                                                     forward_only=True)
        eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
        print("  eval perplexity %.2f" % eval_ppx)
        if current_step % 1000 == 0:
          step_path = checkpoint_data_path + 'step_' + str(current_step) + '.pkl'
          if not os.path.exists(checkpoint_data_path):
            os.makedirs(checkpoint_data_path)
          with open(step_path, 'wb') as step_file:
            Pickle.dump((attns, input_data, target_data, output_logits), step_file, protocol=-1)
        sys.stdout.flush()


def decode(interactive=False):
  # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
  print("Forward pass...")
  config = tf.ConfigProto(allow_soft_placement = True)
  train_batches, dev_batches = data_ops.load_batches(FLAGS.batch_size)
  encoder_counts, encoder_ids, decoder_counts, decoder_ids  = data_ops.load_vocabularies(
    FLAGS.encoder_vocab_size, FLAGS.decoder_vocab_size)

  def divide_list(l, n):
    for i in range(0, len(l), n):
      yield l[i:i + n]
  
  with tf.Session(config=config) as sess:
    # Create model and load parameters.
    model = create_model(sess, forward_only=True)

    
    info = []
    for devidx, dev_batch in enumerate(dev_batches):
      # if devidx % 10 == 0:
      # print('dev step: %s' % devidx)
      sys.stdout.flush()
      input_data, target_data, target_weights = model.get_batch(next(dev_batches), 
                                                                reverse_input=FLAGS.reverse_input)
      eval_loss, attns, output_logits = model.step(sess, input_data, target_data, 
                                                   target_weights, 
                                                   forward_only=True)
      # seq_entropy = data_ops.seq_entropy(input_data)
      output_data = np.array(output_logits)
      input_data = np.array(input_data)
      target_data = np.array(target_data)
      attn_data = np.squeeze(np.array(attns))

      sample_idx = 0
      sample_input = input_data[sample_idx]
      sample_output = output_data[sample_idx]
      sample_valid = target_data[sample_idx]
      sample_attns = attn_data[sample_idx]
      current_batch = 0
      batch_text = []
      # inpt =  {'sen': 10, 'word': 40} # text 10 sen, 40 words per sen  
      # otpt =  {'sen': 5, 'word': 8} # category 5 sen, 8 words per sen

      for idx in range(FLAGS.batch_size):
        print('\n--------------------- batch %s:%s -------------------------\n' % (devidx, idx))        
        input_ids = input_data[:,idx]
        input_attns = attn_data[:,idx,:]
        input_words = [encoder_ids[x][0] for x in input_ids.tolist()]
        # input_textdata = list(zip(input_words, input_attns))        
        # input_text, _ = data_ops.get_textdata(input_textdata)
        input_text = data_ops.get_text(input_words)
        target_ids = target_data[:,idx]
        target_words = [decoder_ids[x][0] for x in target_ids]
        target_words = [x if x != '_EOS' else '\n' for x in target_words]
        target_words = [x for x in target_words if x not in data_ops._START_VOCAB]
        target_words = ', '.join(target_words).replace(', \n, ', '\n')
        target_text = '\n'.join([x.strip() for x in target_words.splitlines() if x.strip()])
        output_ids = output_data[:, idx, :]
        output_maxes = [int(np.argmax(logits)) for logits in output_ids]
        print(np.array(decoder_ids).shape)
        exit(1)
        output_words = [decoder_ids[x][0] for x in output_maxes]
        output_words = list(divide_list(output_words,8))
        output_words = [[x for x in y if x not in data_ops._START_VOCAB] for y in output_words]
        output_text = '\n'.join([' '.join([x.strip() for x in y if x.strip()]) for y in output_words])
        # output_words = [x for x in output_words if x not in data_ops._START_VOCAB]
        # output_words = list(uniq(output_words))
        # output_trunc = []
        # for sen in output_words:
        #   if sen[0] == 'EOS':
        #     output_trunc.append('\n')
        #     continue
        #   else: output_trunc.append(sen)
        # output_text = ', '.join(output_words)
        print('%s\n\n%s\n\n%s' % (input_text.replace('\n', ''), 
                                  colored(target_text, 'green'),
                                  colored(output_text, 'blue')))

        # exit(1)
        # words = (input_words, target_words, output_words)
        # text = (input_text.replace('\n', ''), target_text, output_text.split('\n')[0])
        # batch_text.append((words, text))
        # info.append((target_text, output_text))
      exit(1)
    # references, summaries = zip(*info)
    # data_ops.summarization_scores(references, summaries)
    batch_path = model_data_path + '/hypotheses_summaries_reverse_dataset.pkl'
    with open(batch_path, 'wb') as batch_file:
      Pickle.dump(info, batch_file, protocol=-1)

      
def self_test():
  e_size =  {'h1': 3, # encoder level 1
             'h2': 2} # encoder level 2
  
  d_size =  {'h1': 3, # decoder level 1
             'h2': 2}# decoder level 2
  
  model_size = {'encoder': e_size, 'decoder': d_size}
  
  with tf.Session() as sess:
    model = hs2s_model.HS2S(
      source_vocab_size=10,
      target_vocab_size=10,
      model_size=model_size,
      lstm_size=10,
      num_layers=4,
      dropout=0.7,
      max_gradient_norm=5,
      batch_size=7,
      learning_rate=0.1,
      learning_rate_decay_factor=0.99,
      num_samples=0)

    sess.run(tf.initialize_all_variables())
    
    data_set ={
      '1': [([1, 2, 3],[3, 2, 1]),   # input
            ([3, 2, 1],[1, 2, 3])],  # target
      '2': [([2, 3, 4],[4, 3, 2]),   # input
            ([4, 3, 2],[2, 3, 4])],  # target
      '3': [([3, 4, 5],[5, 4, 3]),   # input
            ([5, 4, 3],[3, 4, 5])],  # target
      '4': [([4, 5, 6],[6, 5, 4]),   # input
            ([6, 5, 4],[4, 5, 6])],  # target
      '5': [([5, 6, 7],[7, 6, 5]),   # input
            ([7, 6, 5],[5, 6, 7])],  # target
      '6': [([6, 7, 8],[8, 7, 6]),   # input
            ([8, 7, 6],[6, 7, 8])],  # target
      '7': [([7, 8, 9],[9, 8, 7]),   # input
            ([9, 8, 7],[7, 8, 9])],  # target
    } 
    encoder_inputs_raw = [v[0] for k,v in list(data_set.items())]
    decoder_inputs_raw = [v[1] for k,v in list(data_set.items())]
    batch = encoder_inputs_raw, decoder_inputs_raw

    # Train the test model for 20000 steps
    step_time, loss = 0.0, 0.0
    for k in range(20000):
      start_time = time.time()
      input_data, target_data, target_weights = model.get_batch(batch, self_test=True)
      # step_loss, step_attn, step_logits
      step_loss = model.step(
        sess, input_data, target_data, target_weights, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      if k % 100 == 0:
        loss, attn, logits = model.step(sess, input_data, target_data, target_weights, 
                                  forward_only=True)
        preds = np.array([l.argmax(axis=1) for l in logits])
        print('\nstep {} loss: {}'.format(k, loss))        
        print('input:\n{}'.format(np.array(input_data)))
        print('target:\n{}'.format(np.array(target_data)[3:,:]))        
        print('output:\n{}\n'.format(preds))

def main(_):
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    with tf.device('/device:GPU:1'):
      decode()
  elif FLAGS.compute_entropy:
    seq_entropy()
  else:
    train()

if __name__ == "__main__":
 tf.app.run()
