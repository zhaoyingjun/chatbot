#导入依赖包
import tensorflow as tf
from config import getConfig
import horovod.tensorflow as hvd
tf.config.experimental_run_functions_eagerly(True)
hvd.init()
#初始化超参字典
gConfig = {}
gConfig= getConfig.get_config()
#通过超参字典为vocab_inp_size、vocab_tar_size、embedding_dim、units等赋值
vocab_inp_size = gConfig['vocab_inp_size']
vocab_tar_size = gConfig['vocab_tar_size']
embedding_dim=gConfig['embedding_dim']
units=gConfig['layer_size']
BATCH_SIZE=gConfig['batch_size']*hvd.size()

#定义Encoder类
class Encoder(tf.keras.Model):
  #初始化函数，对默认参数进行初始化
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
    super(Encoder, self).__init__()
    self.enc_units = enc_units
    self.batch_size = batch_size
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
  #定义调用函数，实现逻辑计算
  def call(self, x, hidden):
    x_emb = self.embedding(x)
    output, state = self.gru(x_emb, initial_state = hidden)
    return output, state
  def initialize_hidden_state(self):
    return tf.zeros((self.batch_size, self.enc_units))
 
#定义bahdanauAttention类，bahdanauAttention是常用的attention实现方法之一
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    #注意力网络的初始化
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)
  def call(self, query, values):
    #将query增加一个维度，以便可以与values进行线性相加
    hidden_with_time_axis = tf.expand_dims(query, 1)
    #将quales与hidden_with_time_axis进行线性相加后，使用tanh进行非线性变换，最后输出一维的score
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))
    #使用softmax将score进行概率化转换，转为为概率空间
    attention_weights = tf.nn.softmax(score, axis=1)
    #将权重与values（encoder_out)进行相乘，得到context_vector
    context_vector = attention_weights * values
    #将乘机后的context_vector按行相加，进行压缩得到最终的context_vector
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    #初始化batch_sz、dec_units、embedding 、gru 、fc、attention
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)
    self.attention = BahdanauAttention(self.dec_units)
  def call(self, y, hidden, enc_output):
    #首先对enc_output、以及decoder的hidden计算attention，输出上下文语境向量
    context_vector, attention_weights = self.attention(hidden, enc_output)
    #对decoder的输入进行embedding
    y = self.embedding(y)
    #拼接上下文语境与decoder的输入embedding，并送入gru中
    y = tf.concat([tf.expand_dims(context_vector, 1), y], axis=-1)
    output, state = self.gru(y)
    #将gru的输出进行维度转换，送入全连接神经网络 得到最后的结果
    output = tf.reshape(output, (-1, output.shape[2]))
    y = self.fc(output)
    return y, state, attention_weights
#定义损失函数
def loss_function(real, pred):
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  #mask掉start,去除start对于loss的干扰
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)#将bool型转换成数值
  loss_ *= mask
  return tf.reduce_mean(loss_)
#实例化encoder、decoder、optimizer、checkpoint等
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()
checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)
@tf.function
def training_step(inp, targ, targ_lang,enc_hidden,first_batch,allreduce=True):
  loss = 0
  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['start']] * BATCH_SIZE, 1)
    for t in range(1, targ.shape[1]):
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
      loss += loss_function(targ[:, t], predictions)
      dec_input = tf.expand_dims(targ[:, t], 1)
  step_loss = (loss / int(targ.shape[1]))
  variables = encoder.trainable_variables + decoder.trainable_variables
  if allreduce:
     tape = hvd.DistributedGradientTape(tape)
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  if first_batch:
    hvd.broadcast_variables(variables, root_rank=0)
    hvd.broadcast_variables(optimizer.variables(), root_rank=0)
  return step_loss





