# -*- coding:utf-8 -*-
import io
import os
import sys
import time
import torch
from torch import optim
from bh_agrt import seq2seqModel
from bh_agrt.config import getConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gConfig = getConfig.get_config()
units = gConfig['layer_size']
BATCH_SIZE = gConfig['batch_size']
EOS_token = 1
SOS_token = 0
MAX_LENGTH = gConfig['max_length']

def preprocess_sentence(w):
    w = 'start ' + w + ' end'
    return w

def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]

    input_lang = Lang("ask")
    output_lang = Lang("ans")
    pairs = [list(reversed(p)) for p in pairs]

    for pair in pairs:
        input_lang.addSentence(pair[1])
        output_lang.addSentence(pair[0])

    return input_lang, output_lang, pairs

def max_length(tensor):
    return max(len(t) for t in tensor)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"start": 0, "end": 1}
        self.word2count = {"start": 0, "end": 0}
        self.index2word = {0: "start", 1: "end"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)

    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def read_data(path, num_examples):
    input_tensors = []
    target_tensors = []
    input_lang, target_lang, pairs = create_dataset(path, num_examples)
    for i in range(0, num_examples - 1):
        input_tensor = tensorFromSentence(input_lang, pairs[i][1])
        target_tensor = tensorFromSentence(target_lang, pairs[i][0])
        input_tensors.append(input_tensor)
        target_tensors.append(target_tensor)
    return input_tensors, input_lang, target_tensors, target_lang

input_tensor, input_lang, target_tensor, target_lang= read_data(
    gConfig['seq_data'], gConfig['max_train_data_size'])
hidden_size = 256

def train():
    print("Preparing data in %s" % gConfig['data'])
    current_epoch = 0
    n_batch = len(input_tensor) // gConfig['batch_size']
    checkpoint_dir = gConfig['model_data']
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, "model.pt")
    start_time = time.time()
    encoder = seq2seqModel.Encoder(input_lang.n_words, hidden_size).to(device)
    decoder = seq2seqModel.AttentionDecoder(hidden_size, target_lang.n_words, dropout_p=0.1).to(device)
    if os.path.exists(checkpoint_prefix):
        checkpoint = torch.load(checkpoint_prefix)
        encoder.load_state_dict(checkpoint['modelA_state_dict'])
        decoder.load_state_dict(checkpoint['modelB_state_dict'])
    max_data = gConfig['max_train_data_size']
    batch_loss = 1
    while batch_loss > gConfig['min_loss']:
        total_loss = 0
        start_time_epoch = time.time()
        for i in range(1, (max_data // BATCH_SIZE)):
            inp = input_tensor[(i - 1) * BATCH_SIZE : i * BATCH_SIZE]
            targ = target_tensor[(i - 1) * BATCH_SIZE : i * BATCH_SIZE]
            batch_loss = seq2seqModel.train_step(inp, targ, encoder, decoder, optim.SGD(encoder.parameters(), lr=0.01, momentum=0.9),
                                                 optim.SGD(decoder.parameters(), lr=0.01, momentum=0.9))
            total_loss += batch_loss
            print('训练batch数:{} 最新batch的loss {:.4f}'.format(i, batch_loss))
        avg_batch_loss = total_loss / n_batch
        current_epoch += 1
        time_per_batch = (time.time() - start_time_epoch) / n_batch
        time_per_epoch = (time.time() - start_time) / current_epoch
        print('训练总epoch数: {} 每epoch平均耗时: {}  每batch平均耗时: {} 该epoch每batch平均loss {:.4f}'.format(
            current_epoch, time_per_epoch, time_per_batch, avg_batch_loss))
        torch.save({'modelA_state_dict': encoder.state_dict(),
                    'modelB_state_dict': decoder.state_dict()}, checkpoint_prefix)
        sys.stdout.flush()


def predict(sentence):
    max_length_tar = MAX_LENGTH
    encoder = seq2seqModel.Encoder(input_lang.n_words, hidden_size).to(device)
    decoder = seq2seqModel.AttentionDecoder(hidden_size, target_lang.n_words, dropout_p=0.1).to(device)
    checkpoint_dir = gConfig['model_data']
    checkpoint_prefix = os.path.join(checkpoint_dir, "model.pt")
    checkpoint=torch.load(checkpoint_prefix)

    encoder.load_state_dict(checkpoint['modelA_state_dict'])
    decoder.load_state_dict(checkpoint['modelB_state_dict'])

    sentence = preprocess_sentence(sentence)
    input_tensor = tensorFromSentence(input_lang, sentence)

    input_length = input_tensor.size()[0]
    result = ''
    max_length = MAX_LENGTH
    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] += encoder_output[0, 0]

    dec_input = torch.tensor([[SOS_token]], device=device)

    dec_hidden = encoder_hidden

    for t in range(max_length_tar):
        predictions, dec_hidden, decoder_attentions = decoder(dec_input, dec_hidden, encoder_outputs)
        predicted_id, topi = predictions.data.topk(1) #
        if topi.item() == EOS_token:
            result += '<EOS>'
            break
        else: #
            result += target_lang.index2word[topi.item()] + ' '
        dec_input = topi.squeeze().detach() #
    result = result.replace('start','').replace('end','').replace(' ','')
    return result


if __name__ == '__main__':
    if len(sys.argv) - 1:
        gConfig = getConfig.get_config(sys.argv[1])
    else:
        gConfig = getConfig.get_config()
    print('\n>> Mode : %s\n' % (gConfig['mode']))

    if gConfig['mode'] == 'train':
        train()
    elif gConfig['mode'] == 'serve':
        print('Serve Usage : >> python3 app.py')
