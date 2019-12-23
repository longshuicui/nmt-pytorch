# -*- coding: utf-8 -*-
"""
@author: longshuicui

@date  :  2019/12/22

@modify:

"""
import re
import random
import torch
import logging
import unicodedata
import torch.nn as nn
from torch import optim as opt
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(name)s %(levelname)s %(message)s")
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#加载数据
logging.info('='*20+'Loading data files'+'>'*5)
SOS_token=0 #开始标记
EOS_token=1 #结束标记

class Lang:
    """生成单词id索引"""
    def __init__(self,name):
        self.name=name
        self.word2index={}
        self.word2count={}
        self.index2word={}
        self.n_words=2  #sos和eos包含在内

    def add_word(self,word):
        if word not in self.word2index:
            self.word2index[word]=self.n_words
            self.word2count[word]=1
            self.index2word[self.n_words]=word
            self.n_words+=1
        else:
            self.word2count[word]+=1

    def add_sentence(self,sentence):
        for word in sentence.split():
            self.add_word(word)

def unicode2ascii(s):
    """将unicode编码变为ascii编码"""
    return ''.join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c) !='Mn')

def normalize_string(s):
    """正则化"""
    s=unicode2ascii(s.lower().strip())
    s=re.sub(r"([.!?])",r" \1",s)
    s=re.sub(r"[^a-zA-Z.?!]+",r" ",s)
    return s

def read_langs(lang1,lang2,reverse=False):
    """读取数据，对应的语言是英语-其他，若将其他语言翻译成英语，reverse设置为True"""
    pairs=[]
    with open('../data/%s-%s.txt'%(lang1,lang2),encoding='utf8') as inp:
        for line in inp.readlines():
            pair=line.strip().split('\t')
            pair=list(map(normalize_string,pair))
            pairs.append(pair)

    if reverse:
        pairs=[list(reversed(p) for p in pairs)]
        input_lang=Lang(lang2)
        output_lang=Lang(lang1)
    else:
        input_lang=Lang(lang1)
        output_lang=Lang(lang2)
    return input_lang,output_lang,pairs

"""
为了提高训练速度，仅仅用短句去做训练，增加文本过滤，选择i am或者he is这样的句子
"""
MAX_LENGTH=10
eng_prefixes=("i am","i m",
              "he is","he s",
              "she is","she s",
              "you are","you re",
              "we are","we re",
              "they are","they re")

def filter_pair(p):
    """条件过滤翻译对"""
    return len(p[0].split())<MAX_LENGTH and len(p[1].split())<MAX_LENGTH and p[0].startswith(eng_prefixes)

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def load_data(lang1='eng',lang2='fra',reverse=False):
    """
    数据加载，包括数据预处理过程：
    读取数据，正则化，生成单词id索引，
    :param lang1: 语言1，默认英语
    :param lang2: 语言2，默认法语
    :param reverse: lang1翻译成lang2，默认False
    :return:
    """
    input_lang,output_lang,pairs=read_langs(lang1,lang2,reverse=reverse)
    logging.info('='*20+'Read %s sentence pairs'%len(pairs)+'>'*5)
    pairs=filter_pairs(pairs)

    logging.info('=' * 20 + 'Trimmed to %s sentence pairs' % len(pairs) + '>' * 5)
    logging.info('=' * 20 + '建立单词id索引' + '>' * 5)

    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    logging.info('=' * 20 + 'lang1:%s, 单词数量:%d'%(input_lang.name,input_lang.n_words) + '>' * 5)
    logging.info('=' * 20 + 'lang2:%s, 单词数量:%d'%(output_lang.name,output_lang.n_words) + '>' * 5)

    return input_lang,output_lang,pairs

###########################################################
#建立seq2seq模型 基于bilstm
class Encoder(nn.Module):
    """
    encoder编码
    input_size: 字典大小
    hidden_size:隐藏层大小，lstm输入维度和隐藏层维度相同
    """
    def __init__(self,input_size,hidden_size):
        super(Encoder,self).__init__()
        self.hidden_size=hidden_size
        self.batch_size=None
        self.embedding=nn.Embedding(num_embeddings=input_size,embedding_dim=hidden_size)

        #batch_first=True输入维度为[batch_size,seq_len,embedding_size],[3,2,3]
        #batch_first=Falae输入维度为[seq_len,batch_size,embedding_size][3,2,3]
        self.lstm=nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers=1,bidirectional=True)

    def forward(self, x,h0,c0):
        embedded=self.embedding(x).view(1,1,-1)
        # output: [seq_len,batch_size,hidden_size*2]
        # h: [directional*num_layers,batch_size,hidden_size]
        # c: [directional*num_layers,batch_size,hidden_size]
        output,hidden=self.lstm(embedded,(h0,c0))
        return output,hidden

    def init_hidden(self):
        return (torch.zeros(2,1,self.hidden_size),
                torch.zeros(2,1,self.hidden_size))

class Decoder(nn.Module):
    """
    decoder解码
    最简单的情况，decoder初始隐藏状态是encoder的最后输出（叫做上下文向量），
    解码过程中的每一步输入是token和上一时刻的隐藏状态，初始时刻输入的是sos和context-vector
    input_size:字典大小
    hidden_size:隐藏向量维度大小
    output_size:字典大小
    """
    def __init__(self,input_size,hidden_size,output_size):
        super(Decoder,self).__init__()
        self.hidden_size=hidden_size
        self.embedding=nn.Embedding(input_size,hidden_size)
        self.lstm=nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,bidirectional=True,num_layers=1)
        self.linear=nn.Linear(hidden_size*2,output_size)
        self.softmax=nn.LogSoftmax(dim=1)

    def forward(self, x, h0, c0):
        embedded=self.embedding(x).view(1,1,-1) #[seq_len,batch_size,embedding_size]
        output=F.relu(embedded)
        output,h=self.lstm(output,(h0, c0))  #[seq_len,batch_size,hidden_size*2]  [num_direction*num*layer,batch_size,hidden_size]
        output=self.linear(output[0]) #[batch_size,output_size]
        output=self.softmax(output) #[batch_size,output_size]
        return output,h

class AttentionDecoder(nn.Module):
    """
    基于注意力机制的解码器
    如果仅仅使用context-vector来传递信息，效果不好。注意力机制可以使decoder关注与encoder中的某一个部分。
    首先计算注意力权重，
    """
    def __init__(self,hidden_size,output_size,dropout=0.1,max_length=MAX_LENGTH):
        super(AttentionDecoder,self).__init__()
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.dropout=dropout
        self.max_length=max_length

        self.embedding=nn.Embedding(num_embeddings=self.output_size,embedding_dim=self.hidden_size)
        self.attention=nn.Linear(self.hidden_size*2,self.max_length)
        self.dropout=nn.Dropout(self.dropout)
        self.att_combine=nn.Linear(self.hidden_size*3,self.hidden_size)
        self.lstm=nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,bidirectional=True) #一层双向 batch_first 默认
        self.out=nn.Linear(self.hidden_size*2,self.output_size)  #线性输出类别

    def forward(self, x,h0,c0,encoder_outputs):
        """前向过程"""
        #embedding
        embed=self.embedding(x).view(1,1,-1) #输入一个样本，每次计算一个seq
        embed=self.dropout(embed)
        #计算注意力权重
        att_score=self.attention(torch.cat([embed[0],h0[0]],-1))
        att_weight=F.softmax(att_score,dim=-1)
        #对encoder的输出做加权求和获得每个时刻的相关结果
        att_applied=torch.bmm(att_weight.unsqueeze(0),encoder_outputs.unsqueeze(0))
        output=torch.cat([embed[0],att_applied[0]],1)
        #对两者相加的结果做一个线性转换
        output=self.att_combine(output)
        output=F.relu(output).unsqueeze(1)
        output,hidden=self.lstm(output,(h0,c0))
        output=self.out(output[0])
        output=F.log_softmax(output,dim=1)
        return output,hidden,att_weight

    def init_hidden(self):
        return (torch.zeros(2,1,self.hidden_size),
                torch.zeros(2,1,self.hidden_size))


###########################################################
#开始模型训练 将输入句子向量化
def index_from_sentence(lang,sentence):
    return [lang.word2index[word] for word in sentence.split()]

def tensor_from_sentence(lang,sentence):
    indexes=index_from_sentence(lang,sentence)
    indexes.append(EOS_token) #增加结束字符
    return torch.tensor(indexes,dtype=torch.long).view(-1,1).to(device) #转换为seq_len,1 的形式


def train(input_tensor=None,
          target_tensor=None,
          encoder=None,
          decoder=None,
          encoder_opt=None,
          decoder_opt=None,
          criterion=None,
          max_length=MAX_LENGTH):
    """
    训练过程
    :param input_tensor: 输入语言向量
    :param target_tensor: 输出语言向量
    :param encoder: 编码器
    :param decoder: 解码器
    :param encoder_opt: 编码器优化器
    :param decoder_opt: 解码器优化器
    :param criterion: 自定义损失函数
    :param max_length: 序列最大长度，默认值为10
    :return:平均损失
    """
    #将优化器梯度置零
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()

    input_length=input_tensor.size(0) #输入向量的序列长度大小
    target_length=target_tensor.size(0) #目标向量的序列长度大小

    encoder_h=encoder.init_hidden()
    encoder_outputs=torch.zeros(max_length,encoder.hidden_size*2,device=device)
    loss=0

    for i in range(input_length):
        #求每一个token的隐藏状态的输出
        encoder_output,encoder_h=encoder(input_tensor[i],encoder_h[0],encoder_h[1])
        encoder_outputs[i]=encoder_output[0,0]

    decoder_input=torch.tensor([[SOS_token]],device=device)
    decoder_h=encoder_h

    #是否根据真实值来预测下一个输出
    use_teacher_forcing=True if random.random()<0.5 else False
    if use_teacher_forcing:
        #用target作为下一时刻的输入
        for di in range(target_length):
            # decoder_output,decoder_h=decoder(decoder_input,decoder_h[0],decoder_h[1])
            # loss+=criterion(decoder_output,target_tensor[di])
            # decoder_input=target_tensor[di]

            #使用注意力的decoder
            decoder_output,decoder_h,att_weight=decoder(decoder_input,decoder_h[0],decoder_h[1],encoder_outputs)
            loss+=criterion(decoder_output,target_tensor[di])
            decoder_input=target_tensor[di]

    else:
        for di in range(target_length):
            # decoder_output,decoder_h=decoder(decoder_input,decoder_h[0],decoder_h[1])
            # topv,topi=decoder_output.topk(1)  #最大的值，最大值的下标
            # decoder_input=topi.squeeze().detach()
            # loss+=criterion(decoder_output,target_tensor[di])
            decoder_output, decoder_h, att_weight = decoder(decoder_input, decoder_h[0], decoder_h[1], encoder_outputs)
            topv,topi=decoder_output.topk(1)
            decoder_input=topi.squeeze().detach()
            loss+=criterion(decoder_output,target_tensor[di])
            if decoder_input.item()==EOS_token:
                break

    loss.backward()
    encoder_opt.step()
    decoder_opt.step()
    return loss.item()/target_length


def evaluate(encoder,decoder,sentence,max_length=MAX_LENGTH):
    """
    模型评估和训练过程一样，只是在decoder中只接受自己的预测值
    :param encoder: 解码器
    :param decoder: 编码器
    :param sentence: 句子
    :param max_length: 最大长度
    :return: 翻译的句子
    """
    with torch.no_grad():
        input_tensor=tensor_from_sentence(input_lang,sentence)
        input_length=input_tensor.size(0)
        encoder_h=encoder.init_hidden()
        encoder_outputs=torch.zeros(max_length,encoder.hidden_size*2,device=device)
        for ei in range(input_length):
            encoder_output,encoder_h=encoder(input_tensor[ei],encoder_h[0],encoder_h[1])
            encoder_outputs[ei]=encoder_output[0,0]

        decoder_input=torch.tensor([[SOS_token]],device=device)
        decoder_h=encoder_h
        decoder_weight=torch.zeros(max_length,max_length)
        decoder_words=[]
        for di in range(max_length):
            decoder_output,decoder_h,decoder_weight_per=decoder(decoder_input,decoder_h[0],decoder_h[1],encoder_outputs)
            decoder_weight[di]=decoder_weight_per.data
            topv,topi=decoder_output.topk(1)
            if topi.item()==EOS_token:
                decoder_words.append('<EOS>')
                break
            else:
                decoder_words.append(output_lang.index2word[topi.item()])
            decoder_input=topi.squeeze().detach()
        return decoder_words,decoder_weight[:di+1]









if __name__ == '__main__':
    n_iters=1000
    input_lang,output_lang,pairs=load_data()
    encoder=Encoder(input_size=input_lang.n_words,hidden_size=64)
    decoder=AttentionDecoder(hidden_size=64,output_size=output_lang.n_words)
    encoder_opt=opt.SGD(encoder.parameters(),lr=0.001)
    decoder_opt=opt.SGD(decoder.parameters(),lr=0.001)

    training_pairs=[]
    for i in range(n_iters):
        pair=random.choice(pairs)
        input_tensor=tensor_from_sentence(input_lang,pair[0])
        output_tensor=tensor_from_sentence(output_lang,pair[1])
        training_pairs.append((input_tensor,output_tensor))

    criterion=nn.NLLLoss() #负对数似然损失
    for k in range(1,n_iters+1):
        #每次迭代只输入一个样本
        input_tensor,output_tensor=training_pairs[k-1]
        loss=train(input_tensor,output_tensor,encoder,decoder,encoder_opt,decoder_opt,criterion)
        print("当前迭代%d，损失为%.4f"%(k,loss))

    #测试
    for i in range(10):
        pair=random.choice(pairs)
        print(pair)
        print('>',pair[0])
        print('=',pair[1])
        output_words,attention=evaluate(encoder,decoder,pair[0])
        output_sentence=' '.join(output_words)
        print('<',output_sentence)
        print()

