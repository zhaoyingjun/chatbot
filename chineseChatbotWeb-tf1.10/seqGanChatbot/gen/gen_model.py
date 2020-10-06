from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import sys
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import utils.data_utils as data_utils
import gen.seq2seq as rl_seq2seq
from tensorflow.python.ops import variable_scope
sys.path.append('../utils')

"""
embedding_attention_seq2seq



model_with_buckets


这两个接和正常的接口是不一样的，我们本次也主要一起看一下这个接口代码有什么不一样的地方

其他的代码结构以及知识点和我们第二课的内容几乎一样，但是我们还是一起过一下

"""



class Seq2SeqModel(object):

    def __init__(self, config, name_scope, forward_only=False, num_samples=512, dtype=tf.float32):
        source_vocab_size = config.vocab_size
        target_vocab_size = config.vocab_size
        emb_dim = config.emb_dim

        self.buckets = config.buckets
        self.learning_rate = tf.Variable(float(config.learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * config.learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.batch_size = config.batch_size
        self.num_layers = config.num_layers
        self.max_gradient_norm = config.max_gradient_norm
        self.mc_search = tf.placeholder(tf.bool, name="mc_search")
        self.forward_only = tf.placeholder(tf.bool, name="forward_only")
        self.up_reward = tf.placeholder(tf.bool, name="up_reward")
        self.reward_bias = tf.get_variable("reward_bias", [1], dtype=tf.float32)
        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < target_vocab_size:
            w_t = tf.get_variable("proj_w", [target_vocab_size, emb_dim], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [target_vocab_size], dtype=dtype)
            output_projection = (w, b)
            w_t=tf.cast(w_t,tf.float32)
            b=tf.cast(b,tf.float32)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
               
                labels=tf.cast(labels,tf.float32)
                inputs=tf.cast(inputs,tf.float32)
                
                return tf.nn.sampled_softmax_loss(w_t ,b, labels,inputs,num_samples,target_vocab_size)
            softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN.
        single_cell = tf.nn.rnn_cell.GRUCell(emb_dim)
        cell = single_cell
        if self.num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return rl_seq2seq.embedding_attention_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                num_encoder_symbols= source_vocab_size,
                num_decoder_symbols= target_vocab_size,
                embedding_size= emb_dim,
                output_projection=output_projection,
                feed_previous=do_decode,
                mc_search=self.mc_search,
                dtype=tf.float32)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(self.buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
        for i in xrange(self.buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[None], name="weight{0}".format(i)))
        self.reward = [tf.placeholder(tf.int32, name="reward_%i" % i) for i in range(len(self.buckets))]

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

        self.outputs, self.losses, self.encoder_state = rl_seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs, targets, self.target_weights,
            self.buckets, source_vocab_size, self.batch_size,
            lambda x, y: seq2seq_f(x, y, tf.where(self.forward_only, True, False)),
            output_projection=output_projection, softmax_loss_function=softmax_loss_function)

        for b in xrange(len(self.buckets)):
            self.outputs[b] = [
                tf.cond(
                    self.forward_only,
                    lambda: tf.matmul(output, output_projection[0]) + output_projection[1],
                    lambda: output
                )
                for output in self.outputs[b]
            ]
        
        if not forward_only:
            with tf.name_scope("gradient_descent"):
                self.gradient_norms = []
                self.updates = []
                self.aj_losses = []
                self.gen_params = [p for p in tf.trainable_variables() if name_scope in p.name]
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                #opt = tf.train.AdamOptimizer()
                for b in xrange(len(self.buckets)):
                    #R =  tf.sub(self.reward[b], self.reward_bias)
                    self.reward[b] = tf.cast(tf.subtract(tf.cast(self.reward[b],tf.float32) , self.reward_bias),tf.int32)
                    

                    #tf.cond 条件判断语句
                    adjusted_loss = tf.cond(self.up_reward,
                                              lambda:tf.multiply(self.losses[b], tf.cast(self.reward[b],tf.float32)),
                                              lambda: self.losses[b])

                    # adjusted_loss =  tf.cond(self.up_reward,
                    #                           lambda: tf.mul(self.losses[b], R),
                    #                           lambda: self.losses[b])
                    self.aj_losses.append(adjusted_loss)
                    gradients = tf.gradients(adjusted_loss, self.gen_params)
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                    self.gradient_norms.append(norm)
                    self.updates.append(opt.apply_gradients(
                        zip(clipped_gradients, self.gen_params), global_step=self.global_step))

        self.gen_variables = [k for k in tf.global_variables() if name_scope in k.name]
        self.saver = tf.train.Saver(self.gen_variables)

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only=True, reward=1, mc_search=False, up_reward=False, debug=True):
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                         " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                         " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.

        input_feed = {
            self.forward_only.name: forward_only,
            self.up_reward.name:  up_reward,
            self.mc_search.name: mc_search
        }
        for l in xrange(len(self.buckets)):
            input_feed[self.reward[l].name] = reward
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only: # normal training
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                       self.aj_losses[bucket_id],  # Gradient norm.
                       self.losses[bucket_id]]  # Loss for this batch.
        else: # testing or reinforcement learning
            output_feed = [self.encoder_state[bucket_id], self.losses[bucket_id]]  # Loss for this batch.
            for l in xrange(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

    

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], outputs[0]  # Gradient norm, loss, no outputs.
        else:
            return outputs[0], outputs[1], outputs[2:]  # encoder_state, loss, outputs.

    def get_batch(self, train_data, bucket_id, batch_size, type=0):

        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # pad them if needed, reverse encoder inputs and add GO to decoder.
        batch_source_encoder, batch_source_decoder = [], []
        # print("bucket_id: %s" %bucket_id)
        if type == 1:
            batch_size = 1
        for batch_i in xrange(batch_size):
            if type == 1:
                encoder_input, decoder_input = train_data[bucket_id]
            elif type == 2:
                # print("disc_data[bucket_id]: ", disc_data[bucket_id][0])
                encoder_input_a, decoder_input = train_data[bucket_id][0]
                encoder_input = encoder_input_a[batch_i]
            elif type == 0:
                encoder_input, decoder_input = random.choice(train_data[bucket_id])
                # print("train en: %s, de: %s" %(encoder_input, decoder_input))

            batch_source_encoder.append(encoder_input)
            batch_source_decoder.append(decoder_input)
            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the disc_data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(batch_size, dtype=np.float64)
            for batch_idx in xrange(batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        return (batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_source_encoder, batch_source_decoder)