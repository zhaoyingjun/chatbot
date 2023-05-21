from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import getConfig
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gConfig= getConfig.get_config()
#SOS 语句开始标志，EOS 语句的结束标志
SOS_token = 0
EOS_token = 1
MAX_LENGTH=gConfig['max_length']
units=gConfig['layer_size']
BATCHSIZE=gConfig['batch_size']
criterion = nn.NLLLoss()
#定义Encoder方法类，用于提取输入语句的特征
class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(Encoder, self).__init__()
    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(input_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size)
  def forward(self, input, hidden):
    #对输入的序列进行embdedding处理
    embedded = self.embedding(input).view(1, 1, -1)
    #在进行embedding处理之后，作为gru网络的输入，输入到gru，提取输入语句的特征。
    output, hidden = self.gru(embedded, hidden)
    return output, hidden
  def initHidden(self):
    return torch.zeros(1, 1, self.hidden_size, device=device)
#定义Decoder方法类，这里的decoder过程是加上了attention机制
class AttentionDencoder(nn.Module):
  def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
    super(AttentionDencoder, self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.dropout_p = dropout_p
    self.max_length = max_length

    self.embedding = nn.Embedding(self.output_size, self.hidden_size)
    self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
    self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.dropout = nn.Dropout(self.dropout_p)
    self.gru = nn.GRU(self.hidden_size, self.hidden_size)
    self.out = nn.Linear(self.hidden_size, self.output_size)
  def forward(self, input, hidden, encoder_outputs):
    embedded = self.embedding(input).view(1, 1, -1)
    embedded = self.dropout(embedded)
    #使用softmax方法来计算出attention的权重值
    attn_weights = F.softmax(
    self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
    attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

    output = torch.cat((embedded[0], attn_applied[0]), 1)
    output = self.attn_combine(output).unsqueeze(0)
    output = F.relu(output)
    output, hidden = self.gru(output, hidden)
    output = F.log_softmax(self.out(output[0]), dim=1)
    return output, hidden, attn_weights
  def initHidden(self):
    return torch.zeros(1, 1, self.hidden_size, device=device)
def min(a,b):
  if a>b:
    return b
  else:
    return a
#定义训练方法
def train_step(input_tensors, target_tensors,encoder,decoder,encoder_optimizer, decoder_optimizer):
  encoder_hidden = encoder.initHidden()
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()
  loss = 0
  for i in range(BATCHSIZE):
    input_tensor=input_tensors[i]
    target_tensor = target_tensors[i]
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    lenth_min=min(MAX_LENGTH,input_length)
    for ei in range(lenth_min):
       encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
       encoder_outputs[ei] = encoder_output[0, 0]
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
              decoder_input, decoder_hidden, encoder_outputs)
        loss += (criterion(F.log_softmax(decoder_output,dim=1), target_tensor[di]))/BATCHSIZE
        decoder_input = target_tensor[di]
  loss.backward()
  encoder_optimizer.step()
  decoder_optimizer.step()
  return loss.item() / target_length







