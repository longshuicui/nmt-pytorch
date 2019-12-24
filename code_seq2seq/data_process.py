# -*- coding: utf-8 -*-
"""
@author: longshuicui

@date  :  2019/12/23

@modify:

"""
import os
import re
import random
import time
import logging
import unicodedata
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(name)s %(levelname)s %(message)s")

class Lang:
    """生成word-index映射字典"""
    def __init__(self,name):
        self.name=name
        self.word2index={'<PAD>':0, '<UNK>':1, '<SOS>':2, '<EOS>':3}
        self.index2word={0:'<PAD>', 1:'<UNK>', 2:'<SOS>', 3:'<EOS>'}
        self.n_words=len(self.word2index)
        self.word2count={}

    def add_word(self,word):
        if word not in self.word2index:
            self.word2count[word]=self.word2count.get(word,0)+1
            self.word2index[word]=self.n_words
            self.index2word[self.n_words]=word
            self.n_words += 1
        else:
            self.word2count[word]=self.word2count.get(word,0)+1

    def add_sentence(self,sentence):
        for word in sentence.split():
            self.add_word(word)

def normalize(sentence):
    """
    数据预处理：
        1.全部改为小写，
        2.正则化过滤多余字符
        3.将缩写改为全拼这个先不写，法语的缩写不太清楚
        4.将unicode编码改为ascii编码
    :param sentence:
    :return:
    """
    # eng_prefixes={"i am":"i m",
    #               "i will": "i ll",
    #               "he is":"he s",
    #               "she is":"she s",
    #               "you are":"you re",
    #               "we are":"we re",
    #               "they are":"they re"}
    sentence=''.join(c for c in unicodedata.normalize('NFD', sentence) if unicodedata.category(c) != 'Mn')
    sentence=sentence.lower()
    sentence=re.sub(r"([.?!])",r" \1",sentence) #将标点符号与字母之间空一格
    sentence=re.sub(r"[^a-zA-Z.?!]+", r" ", sentence) #将非字母和标点的替换成空格
    return sentence

def load_data(source='eng',target='fra'):
    """
    读取数据
    :param source: source翻译语言，默认英语
    :param target: object目标语言，默认法语
    :return: 语言对 [src,obj]
    """
    logging.info("=" * 10 + 'loading data....')
    pairs=[]
    with open('../data/%s-%s.txt'%(source,target),encoding='utf8') as file:
        for line in file:
            pair=line.strip().split('\t')
            pairs.append(pair)

    logging.info("=" * 10 + '样本数量%d' % (len(pairs)))
    logging.info("=" * 10 + '数据预处理')

    pairs=list(map(lambda x:[normalize(x[0]),normalize(x[1])],pairs))
    #统计文本最长的长度和最短的长度
    src_len=list(map(lambda x:len(x[0]),pairs))
    tar_len=list(map(lambda x:len(x[1]),pairs))
    src_max_len,src_min_len=max(src_len),min(src_len)
    tar_max_len,tar_min_len=max(tar_len),min(tar_len)

    logging.info("="*10+'src最大长度%d,最小长度%d'%(src_max_len,src_min_len))
    logging.info("="*10+'tar最大长度%d,最小长度%d'%(tar_max_len,tar_min_len))
    logging.info("=" * 10 + '生成word-index映射字典')

    #统计一下样本各个分区的数量，便于之后的分桶 [0,50],[51,100],[101,-1]
    src_lang=Lang(name=source)
    tar_lang=Lang(name=target)
    for pair in pairs:
        src_lang.add_sentence(pair[0])
        tar_lang.add_sentence(pair[1])

    return src_lang, tar_lang, pairs

def sentence2tensor(lang,sentence):
    """
    将单条文本转化为id
    :param lang: 输入的语言
    :param sentence:文本数据
    :return: 数值向量
    """
    return [lang.word2index[w] for w in sentence.split()]

def text2tensor(src_lang, tar_lang, pairs):
    """
    批量转换成id向量
    :param src_lang: 输入语言
    :param tar_lang: 目标语言
    :param pairs: 文本对
    :return: ids
    """
    logging.info("=" * 10 + 'transfer text to vector....')
    vectors=[]
    for pair in pairs:
        vectors.append([sentence2tensor(src_lang,pair[0]),sentence2tensor(tar_lang,pair[1])])
    for i in range(5):
        pair=pairs[i]
        logging.info("=" * 10 + 'text:%s'%str(pair)+'  ==>  '+'vector:%s'%str(vectors[i]))
    return vectors











if __name__ == '__main__':
    #1.加载数据 与 属于预处理
    src_lang, tar_lang, pairs=load_data()
    #2.数据ids化
    vectors=text2tensor(src_lang,tar_lang,pairs)