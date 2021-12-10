import tensorflow as tf
import numpy as np
import os
import time
import datetime
import random
import utils.data_utils as data_utils
from disc.rnnModel import rnnModel
from tensorflow.python.platform import gfile
import sys
from six.moves import xrange

sys.path.append("../utils")


def evaluate(session, model, config, evl_inputs, evl_labels, evl_masks):
    total_num = len(evl_inputs[0])

    fetches = [model.correct_num, model.prediction, model.logits, model.target]
    feed_dict = {}
    for i in xrange(config.max_len):
        feed_dict[model.input_data[i].name] = evl_inputs[i]
    feed_dict[model.target.name] = evl_labels
    feed_dict[model.mask_x.name] = evl_masks
    correct_num, prediction, logits, target = session.run(fetches, feed_dict)

    print("total_num: ", total_num)
    print("correct_num: ", correct_num)
    print("prediction: ", prediction)
    print("target: ", target)

    accuracy = float(correct_num) / total_num
    return accuracy


def hier_read_data(config, query_path, answer_path, gen_path):
    query_set = [[] for _ in config.buckets]
    answer_set = [[] for _ in config.buckets]
    gen_set = [[] for _ in config.buckets]
    with gfile.GFile(query_path, mode="r") as query_file:
        with gfile.GFile(answer_path, mode="r") as answer_file:
            with gfile.GFile(gen_path, mode="r") as gen_file:
                query, answer, gen = query_file.readline(), answer_file.readline(), gen_file.readline()
                counter = 0
                while query and answer and gen:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading disc_data line %d" % counter)
                    query = [int(id) for id in query.strip().split()]
                    answer = [int(id) for id in answer.strip().split()]
                    gen = [int(id) for id in gen.strip().split()]
                    for i, (query_size, answer_size) in enumerate(config.buckets):
                        if len(query) <= query_size and len(answer) <= answer_size and len(gen) <= answer_size:
                            query = query[:query_size] + [data_utils.PAD_ID] * (query_size - len(query) if query_size > len(query) else 0)
                            query_set[i].append(query)
                            answer = answer[:answer_size] + [data_utils.PAD_ID] * (answer_size - len(answer) if answer_size > len(answer) else 0)
                            answer_set[i].append(answer)
                            gen = gen[:answer_size] + [data_utils.PAD_ID] * (answer_size - len(gen) if answer_size > len(gen) else 0)
                            gen_set[i].append(gen)
                    query, answer, gen = query_file.readline(), answer_file.readline(), gen_file.readline()

    return query_set, answer_set, gen_set


def hier_get_batch(config, max_set, query_set, answer_set, gen_set):
    batch_size = config.batch_size
    if batch_size % 2 == 1:
        return IOError("Error")
    train_query = []
    train_answer = []
    train_labels = []
    half_size = batch_size // 2
    for _ in range(half_size):
        index = random.randint(0, max_set)
        train_query.append(query_set[index])
        train_answer.append(answer_set[index])
        #真实数据是正类
        train_labels.append(1)
        train_query.append(query_set[index])
        train_answer.append(gen_set[index])
        #gen产生的是父类
        train_labels.append(0)
    return train_query, train_answer, train_labels


def create_model(sess, config, name_scope, initializer=None):
    with tf.variable_scope(name_or_scope=name_scope, initializer=initializer):
        model = rnnModel(config=config, name_scope=name_scope)
        disc_ckpt_dir = os.path.abspath(os.path.join(config.train_dir, "checkpoints"))
        ckpt = tf.train.get_checkpoint_state(disc_ckpt_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading Hier Disc model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created Hier Disc model with fresh parameters.")
            disc_global_variables = [gv for gv in tf.global_variables() if name_scope in gv.name]
            sess.run(tf.variables_initializer(disc_global_variables))
        return model


def prepare_data(config):
    train_path = os.path.join(config.train_dir, "train")
    voc_file_path = [train_path + ".query", train_path + ".answer", train_path + ".gen"]
    vocab_path = os.path.join(config.train_dir, "vocab%d.all" % config.vocab_size)
    data_utils.create_vocabulary(vocab_path, voc_file_path, config.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    print("Preparing train disc_data in %s" % config.train_dir)
    train_query_path, train_answer_path, train_gen_path =data_utils.hier_prepare_disc_data(config.train_dir, vocab, config.vocab_size)
    query_set, answer_set, gen_set = hier_read_data(config, train_query_path, train_answer_path, train_gen_path)
    return query_set, answer_set, gen_set


def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob


def hier_train(config_disc, config_evl):
    config_evl.keep_prob = 1.0

    print("begin training")

    with tf.Session() as session:

        print ("prepare_data")
        #数据来源：
        query_set, answer_set, gen_set = prepare_data(config_disc)

        train_bucket_sizes = [len(query_set[b]) for b in xrange(len(config_disc.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
        #dev_query_set, dev_answer_set, dev_gen_set = hier_read_data(dev_query_path, dev_answer_path, dev_gen_path)
        for set in query_set:
            print("set length: ", len(set))

        model = create_model(session, config_disc, name_scope=config_disc.name_model)

        step_time, loss = 0.0, 0.0
        current_step = 0
        #previous_losses = []
        step_loss_summary = tf.Summary()
        disc_writer = tf.summary.FileWriter(config_disc.tensorboard_dir, session.graph)

        while current_step<300:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            start_time = time.time()

            b_query, b_answer, b_gen = query_set[bucket_id], answer_set[bucket_id], gen_set[bucket_id]

            train_query, train_answer, train_labels = hier_get_batch(config_disc, len(b_query)-1, b_query, b_answer, b_gen)

            train_query = np.transpose(train_query)
            train_answer = np.transpose(train_answer)

            feed_dict = {}
            for i in xrange(config_disc.buckets[bucket_id][0]):
                feed_dict[model.query[i].name] = train_query[i]
            for i in xrange(config_disc.buckets[bucket_id][1]):
                feed_dict[model.answer[i].name] = train_answer[i]
            feed_dict[model.target.name] = train_labels

            fetches = [model.b_train_op[bucket_id], model.b_logits[bucket_id], model.b_loss[bucket_id], model.target]
            train_op, logits, step_loss, target = session.run(fetches, feed_dict)

            step_time += (time.time() - start_time) / config_disc.steps_per_checkpoint
            loss += step_loss /config_disc.steps_per_checkpoint
            current_step += 1

            if current_step % config_disc.steps_per_checkpoint == 0:

                disc_loss_value = step_loss_summary.value.add()
                disc_loss_value.tag = config_disc.name_loss
                disc_loss_value.simple_value = float(loss)

                disc_writer.add_summary(step_loss_summary, int(session.run(model.global_step)))

                print("logits shape: ", np.shape(logits))

                # softmax operation
                logits = np.transpose(softmax(np.transpose(logits)))

                reward = 0.0
                for logit, label in zip(logits, train_labels):
                    reward += logit[1]  # only for true probility
                reward = reward / len(train_labels)
                print("reward: ", reward)


                print("current_step: %d, step_loss: %.4f" %(current_step, step_loss))


                if current_step % (config_disc.steps_per_checkpoint ) == 0:
                    print("current_step: %d, save_model" % (current_step))
                    disc_ckpt_dir = os.path.abspath(os.path.join(config_disc.train_dir, "checkpoints"))
                    if not os.path.exists(disc_ckpt_dir):
                        os.makedirs(disc_ckpt_dir)
                    disc_model_path = os.path.join(disc_ckpt_dir, "disc.model")
                    model.saver.save(session, disc_model_path, global_step=model.global_step)


                step_time, loss = 0.0, 0.0
                sys.stdout.flush()

