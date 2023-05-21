import jieba

from bh_agrt import execute

def reply(question):
    tokenized_question = " ".join(jieba.cut(question))
    answer = execute.predict(tokenized_question)
    final_answer = answer.replace('_UNK', '').replace('<EOS>','').strip()
    return final_answer
