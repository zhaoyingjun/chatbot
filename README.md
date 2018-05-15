# chatbot
一个可以使用自己语料进行训练的中文聊天机器人，欢迎大家实践交流。

#关于语料的说明

本次训练的语料是从互联网上找到的shooter的训练语料，语料质量很差劲，仅作为演示代码来用，大家可以使用自己的语料
语料下载地址：https://pan.baidu.com/s/1kWYIOVt,将文件下载后放到data目录下。

#代码执行顺序

1、在下载好代码和语料之后，将语料文件放入data目录下。
2、按照 数据预处理器（data_utls.py)-->execute.py(执行器)-->web/app.py（预测可视化模块）的顺序执行就可以了。
3、超参配置在seq2seq.ini和seq2seq_sever.ini文件中配置。
4、详细的代码讲解可以参与我的chat文章(http://gitbook.cn/books/5a4a16da1f2e8d585e464f44/index.html)。


#参考代码和文献

http://blog.topspeedsnail.com/archives/10735/comment-page-1#comment-1161。

http://www.easyapple.net/?p=1384&from=singlemessage&isappinstalled=0。

#建议环境

ubuntu14.04
python3.5
tensorflow==1.0.1或者tensorflow-gpu==1.0.1
flask==0.11.1

#关于版权

本代码遵循Apache2.0开源协议，作为一个平台供大家学习和研究交流，不过希望大家fork和给star。

#已更新功能清单:

V1.1:已经增加中文分词，效果是变得更好了。注意在使用分词后，需要增加词典的大小，否则的话会导致词典无法覆盖训练集，导致出现很多的UNK。直接在seq2seq.ini中修改超参数enc_vocab_size和dec_vocab_size的值即可。

#未来要更新的功能Flag：
1、考虑到GAN在实践中取得成绩，在未来三个月内会增加一个基于SeqGan的版本，以增加训练的效果。






