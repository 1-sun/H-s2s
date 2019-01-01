from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import random
import copy
import json

import numpy as np
import tensorflow as tf

import data_ops
import model_ops

class HS2S(object):

  def __init__(self,source_vocab_size, target_vocab_size, model_size,
               lstm_size, num_layers, dropout, max_gradient_norm, batch_size, 
               learning_rate, learning_rate_decay_factor, num_samples=512, 
               forward_only=False, dtype=tf.float32):
    
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.model_size = model_size
    self.batch_size = batch_size
    self.lstm_size = lstm_size
    self.num_samples = num_samples
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.global_step = tf.Variable(0, trainable=False)
    # self.lstm_init = tf.random_uniform_initializer(-0.08,0.08) 

    
    # LSTM cells
    def lstm_cell():
      cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size, state_is_tuple=True) #,
                                          #initializer=self.lstm_init)
      if dropout < 1:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
      if num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
      return cell

    self.cells = dict.fromkeys(
      ['encoder_h1','encoder_h2','decoder_h1','decoder_h2'], lstm_cell())

     # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None

    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.target_vocab_size:
      w_t = tf.get_variable("proj_w", [self.target_vocab_size, lstm_size], dtype=dtype)
      w = tf.transpose(w_t)
      b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
      output_projection = (w, b)
      
      def sampled_loss(inputs, labels, attns=None):
        labels = tf.reshape(labels, [-1, 1])
        # We need to compute the sampled_softmax_loss using 32bit floats to
        # avoid numerical instabilities.
        local_w_t = tf.cast(w_t, tf.float32)
        local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(inputs, tf.float32)
        local_outputs = tf.cast(
          tf.nn.sampled_softmax_loss(
            weights=local_w_t,
            biases=local_b,
            labels=labels,
            inputs=local_inputs,
            num_sampled=num_samples,
            num_classes=self.target_vocab_size),
          dtype)
        return local_outputs
      softmax_loss_function = sampled_loss

    # Learning rate decay
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)

    # The annotate function 
    def hs2s_f(encoder_inputs, decoder_inputs, do_decode):
      return model_ops.hs2s(
          self.encoder_inputs, self.decoder_inputs,
          self.lstm_size, self.cells, self.model_size, 
          self.batch_size, embedding_size=self.lstm_size, 
          num_input_symbols=self.source_vocab_size,
          num_decoder_symbols=self.target_vocab_size,
          feed_previous=do_decode, output_projection=output_projection)

    # Feeds for inputs.
    self.encoder_inputs, self.decoder_inputs = [],[]
    self.target_weights = []

    # input document sizes
    self.inpt = self.model_size['encoder']
    self.encoder_h1 = self.inpt['h1']    
    self.encoder_h2 = self.inpt['h2']
    self.encoder_size = self.encoder_h1 * self.encoder_h2

    # output document sizes
    self.otpt = self.model_size['decoder']    
    self.decoder_h1 = self.otpt['h1']
    self.decoder_h2 = self.otpt['h2'] 
    self.decoder_size = self.decoder_h1 * self.decoder_h2
    
    # create tensors
    for i in range(self.encoder_size):
      self.encoder_inputs.append(
          tf.placeholder(tf.int32, shape=[None],name="encoder{0}".format(i)))
    for i in range(self.decoder_size + self.decoder_h1):
      self.decoder_inputs.append(
        tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
      self.target_weights.append(
        tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))
    
    # targets are decoder inputs shifted by one sentence.
    targets = [self.decoder_inputs[i + self.decoder_h1]
               for i in range(self.decoder_size)]


    # Training outputs and losses.
    if forward_only:
      self.outputs, self.losses, self.attns = model_ops.hs2s_model(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, lambda x, y: hs2s_f(x, y, True),
          softmax_loss_function=softmax_loss_function)
      # If we use output projection, we need to project outputs for decoding.
      if output_projection is not None:
        self.outputs = [tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs]
    else:
      self.outputs, self.losses, self.attns = model_ops.hs2s_model(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, lambda x, y: hs2s_f(x, y, False),
          softmax_loss_function=softmax_loss_function)


    # Compute and apply gradients
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      gradients = tf.gradients(self.losses, params,
                  aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
                  #aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
      self.updates = opt.apply_gradients(zip(clipped_gradients, params), 
                                        global_step=self.global_step)
    # self.cost  = tf.reduce_sum(self.losses)/self.batch_size

    # Model saver
    self.saver = tf.train.Saver(tf.all_variables())


  def step(self, session, encoder_inputs, decoder_inputs, target_weights, forward_only):

    # Input feed: input data, target data, target_weights
    input_feed = {}  
    for l in range(len(encoder_inputs)):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in range(len(decoder_inputs)):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
    for l in range(len(target_weights)):
      input_feed[self.target_weights[l].name] = target_weights[l]

    # # Since our targets are decoder inputs shifted by one sentence, we need one more.
    for i in range(self.decoder_h1):
      last_target = self.decoder_inputs[self.decoder_size + i].name
      input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Don't run optimizer if forward_only
    if not forward_only:
      output_feed = [self.updates, self.losses]
    else:
      output_feed = [self.losses, self.attns]
      for l in range(self.decoder_size):  # Output logits.
        output_feed.append(self.outputs[l])
      
    # Run the model
    outputs = session.run(output_feed, input_feed)      
    if not forward_only:
      return outputs[1]
    else:
      return outputs[0], outputs[1], outputs[2:]

  def get_batch(self, batch, reverse_input=False, print_batch=False, self_test=False):
    encoder_inputs, decoder_inputs = [],[]

    if self_test:
      encoder_inputs_raw, decoder_inputs_raw = batch
      encoder_inputs = encoder_inputs_raw
      # the decoder input needs an initializer "go" symbol
      decoder_inputs = [([0,0,0],h1,h2) for h1,h2 in decoder_inputs_raw]
    else:
      encoder_inputs_raw, decoder_inputs_raw = zip(*batch)
      encoder_inputs_raw = [[[x[1] for x in y] for y in z] for z in encoder_inputs_raw]
      decoder_inputs_raw = [[[x[1] for x in y if x[1] < 3878] for y in z] for z in decoder_inputs_raw]
    
      # Reverse if reverse_input=True, and pad source document
      encoder_empty_sen = [data_ops.PAD_ID] * self.num_encoder_word


      for i, doc in enumerate(encoder_inputs_raw):
        encoder_doc = []
        num_encoder_raw_sen = len(doc)
        num_encoder_pad = self.num_encoder_sen - num_encoder_raw_sen    
        for encoder_sen in list(doc):
          if len(encoder_sen) > self.num_encoder_word:
            encoder_sen = encoder_sen[:self.num_encoder_word]
          encoder_pad = [data_ops.PAD_ID] * (self.num_encoder_word - len(encoder_sen))
          encoder_sen += encoder_pad
          if reverse_input:
            encoder_sen = list(reversed(encoder_sen))
          encoder_doc.append(encoder_sen)
        for _ in range(num_encoder_pad):
          encoder_doc.append(encoder_empty_sen)
        if reverse_input:
          encoder_doc = list(reversed(encoder_doc))
        encoder_inputs.append(encoder_doc)

      # Pad target document
      decoder_empty_sen = [data_ops.PAD_ID] * self.num_decoder_word

      for idx, doc in enumerate(decoder_inputs_raw):
        decoder_doc = []
        num_decoder_raw_sen = len(doc)
        num_decoder_pad = self.num_decoder_sen - num_decoder_raw_sen

        for i, decoder_sen in enumerate(list(doc)):
          print(decoder_sen)
          if i == 0:
            decoder_pad = [data_ops.PAD_ID] * (self.num_decoder_word - len(decoder_sen) - 2)
            decoder_doc.append(list(decoder_sen) + [data_ops.GO_ID] + [data_ops.EOS_ID] + decoder_pad)
          elif i == len(decoder_sen)-1:
            decoder_pad = [data_ops.PAD_ID] * (self.num_decoder_word - len(decoder_sen) - 2)
            decoder_doc.append(list(decoder_sen) + [data_ops.EOS_ID] + [data_ops.EOD_ID] + decoder_pad)
          else:
            decoder_pad = [data_ops.PAD_ID] * (self.num_decoder_word - len(decoder_sen) - 1)
            decoder_doc.append(list(decoder_sen) + [data_ops.EOS_ID] + decoder_pad)

        for _ in range(num_decoder_pad):
          decoder_doc.append(decoder_empty_sen)

        decoder_inputs.append(decoder_doc)

    # Create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights  = [], [], []

    # Batch input data are just re-indexed inputs.
    encoder_inputs_flat = [[w for s in doc for w in s] for doc in encoder_inputs]
    for length_idx in range(self.encoder_size):
      batch_encoder_inputs.append(
        np.array([encoder_inputs_flat[batch_idx][length_idx]
                  for batch_idx in range(self.batch_size)], dtype=np.int32))
      
    # Batch target data are re-indexed targets, we create target_weights.
    # iterate over decoder inputs and allocate weights
    for idx, doc in enumerate(decoder_inputs):
      weight_idx = 0
      doc_weights = np.ones(self.decoder_size, dtype=np.float32)
      flat_doc = [w for s in doc for w in s]
      for length_idx in range(self.decoder_h2):
        for word_idx in range(self.decoder_h1):
          # if PAD then weight is 0
          # The corresponding target is decoder_input shifted by 1 forward.
          if length_idx < self.decoder_h2-1:
            target = doc[length_idx+1][word_idx]
          if length_idx == self.decoder_h2-1 or target == data_ops.PAD_ID:
            doc_weights[weight_idx] = 0.0
          weight_idx += 1
      batch_weights.append(doc_weights)
      batch_decoder_inputs.append(np.array(flat_doc))
    batch_decoder_inputs = [np.array(x) for x in zip(*batch_decoder_inputs)]
    batch_weights = [np.array(x) for x in zip(*batch_weights)]

    # print batch & target weights
    if print_batch:
      for row in batch_encoder_inputs:
        print(row.tolist())
        print('\n')
      for row in batch_decoder_inputs:
        print(row.tolist())
        print('\n')
      for row in batch_weights:
        print(row.tolist())
      sys.exit(1)

    return batch_encoder_inputs, batch_decoder_inputs, batch_weights
