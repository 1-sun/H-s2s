from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn

# from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear as linear
# from tensorflow.contrib.rnn.python.ops.rnn_cell import _Linear as linear
from tensorflow.contrib.rnn.python.ops import core_rnn_cell as rnn_cell
linear = rnn_cell._linear


from tensorflow.python.ops import variable_scope

import tensorflow as tf
import numpy as np

# linear = core_rnn_cell_impl._linear 

def zero_state(h):
  c = tf.zeros_like(h)
  return tf.concat([c, h], 1)

def seq_blocks(l, n):
  for i in range(0, len(l), n):
      yield l[i:i + n]


def _extract_argmax_and_embed(embedding, output_projection=None,
                              update_embedding=True):
  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(
        prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev
  return loop_function 
      

# to decode without attn, just refer up
# the loop function should apply just the same
# return None for attns
def decoder(): None


def attn_decoder(decoder_inputs, attention_states, encoder_state,
                 cells, model_size, lstm_size, batch_size, embedding_size, 
                 num_symbols, loop_function=None, num_heads=1,
                 initial_state_attention=False, output_size=None, 
                 attention=True, scope=None):

  # encoder size
  num_encoder_word = model_size['encoder']['h1']    
  num_encoder_sen = model_size['encoder']['h2'] 
  # decoder size 
  num_decoder_word = model_size['decoder']['h1']  
  num_decoder_sen = model_size['decoder']['h2']
  
  outputs, attn_outputs = [],[]

  if output_size is None:
    output_size = cells["decoder_h1"].output_size

  with variable_scope.variable_scope(scope or "attention_decoder"):
    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    # print(attention_shapes.get_shape())
    attn_length = attention_states.get_shape()[1].value
    attn_size = attention_states.get_shape()[2].value
    word_attn_size = num_encoder_word * num_encoder_sen

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = array_ops.reshape(
        attention_states, [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []
    attention_vec_size = attn_size  # Size of query vectors for attention.
    for a in range(num_heads):
      k = variable_scope.get_variable("AttnW_%d" % a,
                                      [1, 1, attn_size, attention_vec_size])
      # print(k.get_shape())
      hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(variable_scope.get_variable("AttnV_%d" % a,
                                           [attention_vec_size]))

    def attention(state):
      """Put attention masks on hidden using hidden_features and query."""
      if np.array(state).ndim > 1:
        concat_layers = [tf.concat([c,h],1) for c,h in state]
        query = tf.concat(concat_layers,1)
      else:
        query = tf.concat([state[0],state[1]],1)
      ds, ass = [],[]  # Results of attention reads will be stored here.
      for a in range(num_heads):
        with variable_scope.variable_scope("Attention_%d" % a):
          y = linear(query, attention_vec_size, True)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(
              v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
          a = nn_ops.softmax(s)
          ass.append(a)
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(array_ops.reshape(
              a, [-1, attn_length, 1, 1]) * hidden,[1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size]))
      return ds, ass

    batch_attn_size = array_ops.stack([batch_size, attn_size])
    batch_word_attn_size = array_ops.stack([batch_size, word_attn_size])
    attns = [array_ops.zeros(batch_attn_size, dtype=dtypes.float32) for 
             _ in range(num_heads)]
    word_attns = [array_ops.zeros(batch_attn_size, dtype=dtypes.float32) for 
                  _ in range(num_heads)]
    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])
    if initial_state_attention:
      attns, word_attns = attention(initial_state)

    prev = None
    sen_state = encoder_state
    decoder_word_idx = 0
    for i in range(num_decoder_sen):
      if i > 0: variable_scope.get_variable_scope().reuse_variables()

      with tf.variable_scope(scope or "decode_words"):
        word_input, word_output = None, None
        word_state = cells["decoder_h1"].zero_state(batch_size, tf.float32)

        for t in range(num_decoder_word):
          if t > 0: variable_scope.get_variable_scope().reuse_variables()
          word_state =  word_state if t else sen_state 

          # If loop_function is set, we use it instead of decoder_inputs.
          if loop_function is not None and prev is not None:
            with variable_scope.variable_scope("loop_function", reuse=True):
              word_input = loop_function(prev, i)
          else:
            word_input = decoder_inputs[decoder_word_idx]
            decoder_word_idx += 1

          x = linear([word_input] + attns, output_size, True)
          word_output, word_state = cells["decoder_h1"](x, word_state)
          
          if not i and initial_state_attention:
            with variable_scope.variable_scope(
              variable_scope.get_variable_scope(), reuse=True):
              attns, word_attns = attention(word_state)
          else:
            attns, word_attns = attention(word_state)

          with variable_scope.variable_scope("AttnOutputProjection"):
            output = linear([word_output] + attns, output_size, True)
            outputs.append(output)
            attn_outputs.append(word_attns)
            
          if loop_function is not None:
            prev = word_output

      _, sen_state = cells["decoder_h2"](word_output, sen_state)

  return outputs, sen_state, attn_outputs


def embed_attn_decoder(decoder_inputs, attention_states, encoder_state, 
                       cells, model_size, lstm_size, batch_size, embedding_size, 
                       num_decoder_symbols, num_heads=1, feed_previous=False, 
                       update_embedding_for_previous=True, output_projection=None, 
                       initial_state_attention=False, output_size=None, scope=None,
                       dtype=dtypes.float32):

  if output_size is None:
    output_size = cells["decoder_h1"].output_size

  if output_projection is not None:
    proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_decoder_symbols])

  with variable_scope.variable_scope(scope or "embedding_attention_decoder"):
    embedding = variable_scope.get_variable("embedding", 
                                            [num_decoder_symbols, embedding_size])
    # define the loop function for generating outputs from previous state
    # returns the embedding for argmax of softmax over output
    loop_function = _extract_argmax_and_embed(
      embedding, output_projection,
      update_embedding_for_previous) if feed_previous else None
    emb_inp = [embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
  
    return attn_decoder(emb_inp, attention_states, encoder_state, cells, 
                        model_size, lstm_size, batch_size, embedding_size, 
                        num_decoder_symbols, loop_function=loop_function, 
                        num_heads=num_heads, output_size=output_size, 
                        initial_state_attention=initial_state_attention)


def encoder(encoder_inputs, lstm_size, cells, model_size,
            batch_size, embedding_size, num_symbols, dtype, scope):

  inpt = model_size['encoder']
  esize_h1 = inpt['h1']  
  esize_h2 = inpt['h2']
  embedding_cell = tf.contrib.rnn.EmbeddingWrapper(cells["encoder_h1"], 
                                             embedding_classes=num_symbols,
                                             embedding_size=embedding_size)

  # Encode the first tier (words)
  with variable_scope.variable_scope(scope or "encoder_h1"):
    h1_outputs = []
    h1_state = embedding_cell.zero_state(batch_size, tf.float32)
    t = 0
    for s in range(esize_h2):
      for w in range(esize_h1):
        if t > 0: tf.get_variable_scope().reuse_variables()
        ht_w, h1_state = embedding_cell(encoder_inputs[t], h1_state)
        h1_outputs.append(ht_w)
    t += 1
    output_size = embedding_cell.output_size
    top_states = [array_ops.reshape(e, [-1, 1, output_size]) for e in h1_outputs]
    attention_states = array_ops.concat(top_states, 1)

  # Encode the second tier (sentences)
  with variable_scope.variable_scope(scope or "encoder_h2"):
    h2_state = cells["encoder_h2"].zero_state(batch_size, tf.float32)
    for t in range(esize_h2):
      if t > 0: tf.get_variable_scope().reuse_variables()      
      _, h2_state = cells["encoder_h2"](h1_outputs[t], h2_state)

  return h2_state, attention_states


def hs2s(encoder_inputs, decoder_inputs, lstm_size, cells,
         model_size, batch_size, embedding_size, 
         num_input_symbols, num_decoder_symbols, num_heads=1,
         feed_previous=False, output_projection=None,
         initial_state_attention=False, dtype=dtypes.float32, scope=None):

  # run encoder
  encoder_state, attention_states = encoder(encoder_inputs, lstm_size, cells, 
                                            model_size, batch_size, embedding_size,
                                            num_input_symbols, dtype=dtype, scope=scope)
  # run decoder
  output_size = None
  if output_projection is None:
      cell = rnn_cell.OutputProjectionWrapper(cells['decoder_h1'], num_decoder_symbols)
      output_size = num_decoder_symbols

  if isinstance(feed_previous, bool):
    outputs, decoder_state, attns = embed_attn_decoder(decoder_inputs,
                                        attention_states, encoder_state, cells, 
                                        model_size, lstm_size, batch_size, 
                                        embedding_size, 
                                        num_decoder_symbols, num_heads=num_heads,
                                        feed_previous=feed_previous,
                                        output_size=output_size,
                                        output_projection=output_projection, 
                                        initial_state_attention=initial_state_attention)
    return outputs, decoder_state, attns

  # If feed_previous is a Tensor, construct 2 graphs and use cond.
  def decoder(feed_previous_bool):
    reuse = None if feed_previous_bool else True
    with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=reuse):
      outputs, decoder_state, attns = embed_attn_decoder(
                                            decoder_inputs, 
                                            attention_states, 
                                            encoder_state, cells,
                                            model_size, lstm_size, 
                                            batch_size, embedding_size, 
                                            num_decoder_symbols, 
                                            num_heads=num_heads,
                                            feed_previous=feed_previous_bool, 
                                            output_size=output_size, 
                                            output_projection=output_projection, 
                                            initial_state_attention=initial_state_attention, 
                                            update_embedding_for_previous=False)
      return  outputs + [decoder_state], attns

  outputs_and_state, attns = control_flow_ops.cond(feed_previous,
                                            lambda: decoder(True),
                                            lambda: decoder(False))
  return outputs_and_state[:-1], outputs_and_state[-1], attns


def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):

  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.op_scope(logits+targets+weights, name, 'sequence_loss_by_example'):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      # print(logit.get_shape())
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=target)
      else:
        crossent = softmax_loss_function(logit, target)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):

  with ops.op_scope(logits+targets+weights, name, 'sequence_loss'):
    cost = math_ops.reduce_sum(sequence_loss_by_example(
        logits, targets, weights,
        average_across_timesteps=average_across_timesteps,
        softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, dtypes.float32)
    else:
      return cost


def hs2s_model(encoder_inputs, decoder_inputs, targets, weights,  
               hs2s, name=None, softmax_loss_function=None,
               per_example_loss=False):

  tsize = len(targets)
  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  
  with ops.op_scope(all_inputs, name, 'hs2s'):
    outputs, final_state, attns = hs2s(encoder_inputs, decoder_inputs)
    if per_example_loss:
      loss = sequence_loss_by_example(
        outputs, targets, weights[:tsize],
        softmax_loss_function=softmax_loss_function)
    else:
      loss = sequence_loss(
        outputs, targets, weights[:tsize],
        softmax_loss_function=softmax_loss_function)
  return outputs, loss, attns
