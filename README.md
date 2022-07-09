# chatbot
这是一个可以使用自己语料进行训练的中文聊天机器人项目，包含tensorflow.2x版本和pytorch版本，欢迎大家实践交流以及Star。

#单机版训练效果（loss值在0.5左右）：

![image](https://user-images.githubusercontent.com/12986443/178111215-930d9627-2397-44e8-9db2-3a6339b5786f.png)
# ReleaseNote:
V1.0:

1)重新整合了工程架构，实现全工程项目的统一，并正式发布1.0版本；

2）新增了大规模分布式训练版本，依托horovod分布式训练框架；

3）pytorch版本进行了更新，增加batch_size训练模式。
# RoadMap:
V1.1:Update:2022-09-30

1）增加FAQ问答机器人模块，可以实现闲聊与FAQ问答之间的无缝切换；

2）增加大规模分布式训练的pytorch版本，同样依托horovod分布式训练框架；

3）优化pytorch版本的效果和代码，使代码结构更加合理。

V1.2:Update:2022-12-30

1)引入基于 Transformer的预训练模型作为聊天机器人的后台支撑模型，并实现基于自有语料的fine tune。

V1.3:Update:2023-03-30

1）发布SeqGAN版本

# seq2seq版本代码执行顺序
大家可以使用小黄鸡的语料，下载地址https://github.com/zhaoyingjun/chatbot/blob/master/chineseChatbotWeb-tf2.0/seq2seqChatbot/train_data/xiaohuangji50w_nofenci.conv

1）、在下载好代码和语料之后，将语料文件放入train_data目录下，超参配置在config/seq2seq.ini文件中配置，配置正确的语料文件名。

2）、按照数据预处理器（data_utls.py)-->execute.py(执行器)-->app.py（可视化对话模块）的顺序执行就可以了。

3）、大规模分布式训练版本，参照horovod的启动方式 horovodrun -np n -H host1_ip:port,host2_ip:port,hostn_ip:port python3 excute.py
# 建议环境
ubuntu==18.04  
python==3.6  

TF2.X:

tensorflow==2.6.0

flask==0.11.1

horovod==0.24(分布式训练)

Pytorch:

torch==1.11.0

flask==0.11.1

# 参考代码和文献

http://blog.topspeedsnail.com/archives/10735/comment-page-1#comment-1161。
http://www.easyapple.net/?p=1384&from=singlemessage&isappinstalled=0。
https://github.com/zpppy/seqGan_chatbot

# 交流、联系方式

QQ：934389697
