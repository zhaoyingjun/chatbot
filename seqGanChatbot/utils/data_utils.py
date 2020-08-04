# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for downloading disc_data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    # sentence = tf.compat.as_bytes(sentence)
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path_list, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    """Create vocabulary file (if it does not exist yet) from disc_data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: disc_data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each disc_data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from disc_data %s" % (vocabulary_path, data_path_list))
        vocab = {}
        for data_path in data_path_list:
            with gfile.GFile(data_path, mode="r") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  processing line %d" % counter)
                    line = tf.compat.as_bytes(line)
                    tokens = basic_tokenizer(line)
                    for w in tokens:
                        word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with open(vocabulary_path, mode="w") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with open(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary,
                      tokenizer=None, normalize_digits=True):
    """Tokenize disc_data file and turn into token-ids using given vocabulary file.

  This function loads disc_data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the disc_data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
    if not gfile.Exists(target_path):
        print("Tokenizing disc_data in %s" % data_path)
        # print("target path: ", target_path)
        # vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocabulary, tokenizer, normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_chitchat_data(data_dir, vocabulary, vocabulary_size, tokenizer=None):
    """
  """

    train_path = os.path.join(data_dir, "train")
    # dev_path = get_wmt_enfr_dev_set(data_dir)
    dev_path = os.path.join(data_dir, "test")

    answer_train_ids_path = train_path + (".ids%d.answer" % vocabulary_size)
    query_train_ids_path = train_path + (".ids%d.query" % vocabulary_size)
    data_to_token_ids(train_path + ".answer", answer_train_ids_path, vocabulary, tokenizer)
    data_to_token_ids(train_path + ".query", query_train_ids_path, vocabulary, tokenizer)

    # Create token ids for the development disc_data.
    answer_dev_ids_path = dev_path + (".ids%d.answer" % vocabulary_size)
    query_dev_ids_path = dev_path + (".ids%d.query" % vocabulary_size)
    data_to_token_ids(dev_path + ".answer", answer_dev_ids_path, vocabulary, tokenizer)
    data_to_token_ids(dev_path + ".query", query_dev_ids_path, vocabulary, tokenizer)

    return (query_train_ids_path, answer_train_ids_path,
            query_dev_ids_path, answer_dev_ids_path)


def hier_prepare_disc_data(data_dir, vocabulary, vocabulary_size, tokenizer=None):
    """Get WMT disc_data into data_dir, create vocabularies and tokenize disc_data.

  Args:
    data_dir: directory in which the disc_data sets will be stored.
    en_vocabulary_size: size of the English vocabulary to create and use.
    fr_vocabulary_size: size of the French vocabulary to create and use.
    tokenizer: a function to use to tokenize each disc_data sentence;
      if None, basic_tokenizer will be used.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training disc_data-set,
      (2) path to the token-ids for French training disc_data-set,
      (3) path to the token-ids for English development disc_data-set,
      (4) path to the token-ids for French development disc_data-set,
      (5) path to the English vocabulary file,
      (6) path to the French vocabulary file.
  """
    # Get wmt disc_data to the specified directory.
    # train_path = get_wmt_enfr_train_set(data_dir)
    train_path = os.path.join(data_dir, "train")
    # dev_path = get_wmt_enfr_dev_set(data_dir)
    dev_path = os.path.join(data_dir, "test")

    # Create token ids for the training disc_data.
    query_train_ids_path = train_path + (".ids%d.query" % vocabulary_size)
    answer_train_ids_path = train_path + (".ids%d.answer" % vocabulary_size)
    gen_train_ids_path = train_path + (".ids%d.gen" % vocabulary_size)

    data_to_token_ids(train_path + ".query", query_train_ids_path, vocabulary, tokenizer)
    data_to_token_ids(train_path + ".answer", answer_train_ids_path, vocabulary, tokenizer)
    data_to_token_ids(train_path + ".gen", gen_train_ids_path, vocabulary, tokenizer)

    return (query_train_ids_path, answer_train_ids_path, gen_train_ids_path)


def prepare_disc_data(data_dir, vocabulary, vocabulary_size, tokenizer=None):
    train_path = os.path.join(data_dir, "train")
    # dev_path = get_wmt_enfr_dev_set(data_dir)
    dev_path = os.path.join(data_dir, "test")

    # Create token ids for the training data.
    answer_train_ids_path = train_path + (".ids%d.pos" % vocabulary_size)
    query_train_ids_path = train_path + (".ids%d.neg" % vocabulary_size)
    data_to_token_ids(train_path + ".pos", answer_train_ids_path, vocabulary, tokenizer)
    data_to_token_ids(train_path + ".neg", query_train_ids_path, vocabulary, tokenizer)

    # Create token ids for the development data.
    answer_dev_ids_path = dev_path + (".ids%d.pos" % vocabulary_size)
    query_dev_ids_path = dev_path + (".ids%d.neg" % vocabulary_size)
    data_to_token_ids(dev_path + ".pos", answer_dev_ids_path, vocabulary, tokenizer)
    data_to_token_ids(dev_path + ".neg", query_dev_ids_path, vocabulary, tokenizer)

    return (query_train_ids_path, answer_train_ids_path,
            query_dev_ids_path, answer_dev_ids_path)


def prepare_defined_data(data_path, vocabulary, vocabulary_size, tokenizer=None):
    # vocab_path = os.path.join(data_dir, "vocab%d.all" %vocabulary_size)
    # query_vocab_path = os.path.join(data_dir, "vocab%d.query" %query_vocabulary_size)

    answer_fixed_ids_path = data_path + (".ids%d.answer" % vocabulary_size)
    query_fixed_ids_path = data_path + (".ids%d.query" % vocabulary_size)

    data_to_token_ids(data_path + ".answer", answer_fixed_ids_path, vocabulary, tokenizer)
    data_to_token_ids(data_path + ".query", query_fixed_ids_path, vocabulary, tokenizer)
    return (query_fixed_ids_path, answer_fixed_ids_path)


def get_dummy_set(dummy_path, vocabulary, vocabulary_size, tokenizer=None):
    dummy_ids_path = dummy_path + (".ids%d" % vocabulary_size)
    data_to_token_ids(dummy_path, dummy_ids_path, vocabulary, tokenizer)
    dummy_set = []
    with gfile.GFile(dummy_ids_path, "r") as dummy_file:
        line = dummy_file.readline()
        counter = 0
        while line:
            counter += 1
            dummy_set.append([int(x) for x in line.split()])
            line = dummy_file.readline()
    return dummy_set
