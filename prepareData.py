# -*- coding:utf-8 -*-
import math
import os
import random
import getConfig
from tensorflow.python.platform import gfile
import re
# 特殊标记，用来填充标记对话
PAD = "__PAD__"
GO = "__GO__"
EOS = "__EOS__"  # 对话结束
UNK = "__UNK__"  # 标记未出现在词汇表中的字符
START_VOCABULART = [PAD, GO, EOS, UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
# 定义字典生成函数
def create_vocabulary(input_file,output_file):
    vocabulary = {}
    with open(input_file,'r') as f:
         counter = 0
         for line in f:
            counter += 1
            tokens = [word for word in line.split()]
            for word in tokens:
                if word in vocabulary:
                   vocabulary[word] += 1
                else:
                   vocabulary[word] = 1
         vocabulary_list = START_VOCABULART + sorted(vocabulary, key=vocabulary.get, reverse=True)
          # 取前20000个常用汉字
         if len(vocabulary_list) > 10000:
            vocabulary_list = vocabulary_list[:10000]
         print(input_file + " 词汇表大小:", len(vocabulary_list))
         with open(output_file, 'w') as ff:
               for word in vocabulary_list:
                   ff.write(word + "\n")

# 把对话字符串转为向量形式
def convert_to_vector(input_file, vocabulary_file, output_file):
	print('对话转向量...')
	tmp_vocab = []
	with open(vocabulary_file, "r") as f:#读取字典文件的数据，生成一个dict，也就是键值对的字典
		tmp_vocab.extend(f.readlines())
	tmp_vocab = [line.strip() for line in tmp_vocab]
	vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])#将vocabulary_file中的键值对互换，因为在字典文件里是按照{123：好}这种格式存储的，我们需要换成{好：123}格式

	output_f = open(output_file, 'w')
	with open(input_file, 'r') as f:
		for line in f:
			line_vec = []
			for words in line.split():
				line_vec.append(vocab.get(words, UNK_ID))
			output_f.write(" ".join([str(num) for num in line_vec]) + "\n")#将input_file里的中文字符通过查字典的方式，替换成对应的key，并保存在output_file
	output_f.close()


def prepare_custom_data(working_directory, train_enc, train_dec, test_enc, test_dec, enc_vocabulary_size, dec_vocabulary_size, tokenizer=None):

    # Create vocabularies of the appropriate sizes.
    enc_vocab_path = os.path.join(working_directory, "vocab%d.enc" % enc_vocabulary_size)
    dec_vocab_path = os.path.join(working_directory, "vocab%d.dec" % dec_vocabulary_size)
    
    create_vocabulary(train_enc,enc_vocab_path)
    create_vocabulary(train_dec,dec_vocab_path)
   
    # Create token ids for the training data.
    enc_train_ids_path = train_enc + (".ids%d" % enc_vocabulary_size)
    dec_train_ids_path = train_dec + (".ids%d" % dec_vocabulary_size)
    convert_to_vector(train_enc, enc_vocab_path, enc_train_ids_path)
    convert_to_vector(train_dec, dec_vocab_path, dec_train_ids_path)
 

    # Create token ids for the development data.
    enc_dev_ids_path = test_enc + (".ids%d" % enc_vocabulary_size)
    dec_dev_ids_path = test_dec + (".ids%d" % dec_vocabulary_size)
    convert_to_vector(test_enc, enc_vocab_path, enc_dev_ids_path)
    convert_to_vector(test_dec, dec_vocab_path, dec_dev_ids_path)

    return (enc_train_ids_path, dec_train_ids_path, enc_dev_ids_path, dec_dev_ids_path, enc_vocab_path, dec_vocab_path)


# 用于语句切割的正则表达
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
  #将一个语句中的字符切割成一个list，这样是为了下一步进行向量化训练
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def sentence_to_token_ids(sentence, vocabulary, normalize_digits=True):#将输入语句从中文字符转换成数字符号

  words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]

def initialize_vocabulary(vocabulary_path):#初始化字典，这里的操作与上面的48行的的作用是一样的，是对调字典中的key-value
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with open(vocabulary_path, "r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)
