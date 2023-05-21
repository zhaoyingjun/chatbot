# -*- coding:utf-8 -*-

import os
from config import getConfig
import jieba
from zhon.hanzi import punctuation
import re

gConfig = {}
gConfig= getConfig.get_config()

conv_path = gConfig['resource_data']
 
if not os.path.exists(conv_path):
	exit()

convs = []
with open(conv_path, encoding='utf-8') as f:
	one_conv = []
	for line in f:
		line = line.strip('\n').replace('?', '')
		line=re.sub(r"[%s]+" %punctuation, "",line)
		if line == '':
			continue
		if line[0] == gConfig['e']:
			if one_conv:
				convs.append(one_conv)
			one_conv = []
		elif line[0] == gConfig['m']:
			one_conv.append(line.split(' ')[1])

seq = []
for conv in convs:
	if len(conv) == 1:
		continue
	if len(conv) % 2 != 0:
		conv = conv[:-1]
	for i in range(len(conv)):
		if i % 2 == 0:
			conv[i]=" ".join(jieba.cut(conv[i]))
			conv[i+1]=" ".join(jieba.cut(conv[i+1]))
			seq.append(conv[i]+'\t'+conv[i+1])

seq_train = open(gConfig['seq_data'],'w', encoding='utf-8')

for i in range(len(seq)):
   seq_train.write(seq[i]+'\n')
   if i % 1000 == 0:
      print(len(range(len(seq))), '处理进度：', i)
seq_train.close()


