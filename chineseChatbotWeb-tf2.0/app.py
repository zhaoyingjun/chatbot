
# coding=utf-8
from flask import Flask, render_template, request,jsonify
import execute
import time
import threading
import jieba
"""
定义心跳检测函数
"""
def heartbeat():
    print (time.strftime('%Y-%m-%d %H:%M:%S - heartbeat', time.localtime(time.time())))
    timer = threading.Timer(60, heartbeat)
    timer.start()
timer = threading.Timer(60, heartbeat)
timer.start()

"""
ElementTree在 Python 标准库中有两种实现。
一种是纯 Python 实现例如 xml.etree.ElementTree ，
另外一种是速度快一点的 xml.etree.cElementTree 。
 尽量使用 C 语言实现的那种，因为它速度更快，而且消耗的内存更少
"""


app = Flask(__name__,static_url_path="/static") 

@app.route('/message', methods=['POST'])

#"""定义应答函数，用于获取输入信息并返回相应的答案"""
def reply():
#从请求中获取参数信息
    req_msg = request.form['msg']
#将语句使用结巴分词进行分词
    req_msg=" ".join(jieba.cut(req_msg))
    #调用decode_line对生成回答信息
    res_msg = execute.predict(req_msg)
    #将unk值的词用微笑符号袋贴
    res_msg = res_msg.replace('_UNK', '^_^')
    res_msg=res_msg.strip()
    
    # 如果接受到的内容为空，则给出相应的回复
    if res_msg == ' ':
      res_msg = '请与我聊聊天吧'

    return jsonify( { 'text': res_msg } )

"""
jsonify:是用于处理序列化json数据的函数，就是将数据组装成json格式返回

http://flask.pocoo.org/docs/0.12/api/#module-flask.json
"""
@app.route("/")
def index(): 
    return render_template("index.html")
'''
'''
# 启动APP
if (__name__ == "__main__"): 
    app.run(host = '0.0.0.0', port = 8808) 
