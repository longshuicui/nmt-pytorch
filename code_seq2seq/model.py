# -*- coding: utf-8 -*-
"""
@author: longshuicui

@date  :  2019/12/23

@modify:

"""
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self,d_model=512,num_head=8):
        """
        多头自我注意力机制
        :param d_model: 模型维度，默认512
        :param num_head: 多头机制，默认为8
        """
        super(SelfAttention, self).__init__()
        assert d_model%num_head==0
        self.d_model=d_model
        self.num_head=num_head
        self.k=d_model//num_head

    def forward(self, tensor,mask=None,dropout=None):
        b, s, h = tensor.size()
        query = nn.Linear(h, h)(tensor).view(b * self.num_head, s, self.k)  # [b*head,s,k]
        key = nn.Linear(h, h)(tensor).view(b * self.num_head, s, self.k)  # [b*head,s,k]
        value = nn.Linear(h, h)(tensor).view(b * self.num_head, s, self.k)  # [b*head,s,k]

        score = torch.matmul(query, key.transpose(2, 1)).view(b,self.num_head,s,s)  #[b,head,s,s]
        if mask is not None:
            assert mask.dim()==4   #[batch_size,1,1,seq_len]
            score=score.masked_fill(mask==0, -1e20)
        weight=nn.Softmax(dim=-1)(score/math.sqrt(h)) #[b,head,s,s]
        if dropout is not None:
            weight=F.dropout(weight,p=dropout)
        tensor = torch.matmul(weight.view(-1,s,s), value).view(b, s, h)
        return tensor


class Encoder(nn.Module):
    def __init__(self,
                 batch_size=64,
                 seq_len=50,
                 embedding_size=128,
                 hidden_size=128,
                 num_layers=2,
                 src_vocab_size=None):
        """
        编码器初始化 采用双向LSTM网络
        :param batch_size: 参与训练的一个batch的样本数量
        :param seq_len: 序列长度，默认50
        :param embedding_size: 输入向量维度，默认128
        :param hidden_size: 隐藏状态向量维度，默认128
        :param num_layers: lstm层数，默认2
        :param src_vocab_size: 词典大小
        """
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = src_vocab_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.hidden_size,
                            bidirectional=False,
                            num_layers=self.num_layers,
                            batch_first=True)

    def forward(self, x):
        embed = self.embedding(x)
        outputs, hidden = self.lstm(embed)
        return outputs, hidden

    def init_hidden(self):
        return (torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size))


class Decoder(nn.Module):
    def __init__(self,
                 batch_size=64,
                 embedding_size=128,
                 hidden_size=128,
                 num_layers=2,
                 tar_vocab_size=None):
        """
        解码器初始化，这里用的是单向双层的lstm
        :param batch_size: 参与训练的一个batch的样本数量，默认64
        :param seq_len: 序列长度，默认1
        :param embedding_size: 输入向量维度，默认128
        :param hidden_size: 隐藏层向量维度，默认128
        :param num_layers: lstm层数，默认2
        :param tar_vocab_size: 目标语言词典大小
        """
        super(Decoder, self).__init__()
        self.batch_size=batch_size
        self.seq_len=1
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.vocab_size=tar_vocab_size

        self.embedding=nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.embedding_size,padding_idx=0)
        self.lstm=nn.LSTM(input_size=self.embedding_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers)
        self.linear=nn.Linear(in_features=self.hidden_size,out_features=self.vocab_size)
        self.dropout=nn.Dropout(0.1)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self, x, hidden):
        """
        decoder前向计算过程
        :param x: input,[batch_size,seq_len]
        :param hidden: 初始隐藏状态向量，lstm为(h,c)
        :return: 输出概率
        """
        embed=self.embedding(x).view(self.seq_len,self.batch_size,self.embedding_size)  # [seq_len,batch_size,embedding_size]
        embed=self.dropout(embed)
        # [seq_len,batch_size,hidden_size] [num_layers,batch_size,hidden_size]
        output,hidden=self.lstm(embed,(hidden[0],hidden[1]))
        output=self.linear(output) #[seq_len,batch_size,vocab_size]
        output=self.softmax(output.squeeze(0)) #[batch_size,vocab_size]
        return output,hidden


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder=None,
                 decoder=None,
                 criterion=None,
                 teacher_forcing=True,
                 beam_search=False):
        """
        seq2seq模型
        :param encoder: 编码器，单向双层lstm
        :param decoder: 解码器，单向双层lstm
        :param criterion: 损失函数
        :param teacher_forcing: 训练过程中是否用真实值输入，默认True
        :param beam_search: 训练、测试过程中是否用动态规划寻找最优解，默认False   后面实现
        """
        super(Seq2Seq, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.criterion=criterion
        self.teacher_forcing=teacher_forcing
        self.beam_search=beam_search

    def forward(self, src,tar):
        """
        前向计算过程
        :param src: 翻译语言，[batch_size,seq_len]
        :param tar: 目标语言，[batch_size,seq_len]
        :return: 损失
        """
        #计算encoder
        encoder_output,encoder_hidden=self.encoder(src)

        #decoder初始隐层状态向量为encoder的输出
        decoder_hidden=encoder_hidden
        batch_size,seq_len=tar.size()

        decoder_input=torch.ones(1,batch_size,dtype=torch.long)*2 #decoder第一时刻的输入是开始符号
        decoder_outputs=torch.zeros([seq_len,batch_size],dtype=torch.long)
        total_loss=0

        for di in range(seq_len):
            decoder_output,decoder_hidden=self.decoder(decoder_input.long(),decoder_hidden)
            topv, topi = decoder_output.topk(1, dim=-1)  # 概率最大的值和下标，获取下一时刻的值作为输入[batch_size,1]
            decoder_outputs[di]=topi.squeeze(1)
            total_loss+=self.criterion(decoder_output,tar[:,di])
            if self.teacher_forcing and random.random()>0.5:
                #使用真实值作为下一时刻的输入
                decoder_input=tar[:,di]
            else:
                #使用预测值作为下一时刻的输入
                decoder_input=torch.zeros(batch_size,1)
                for i in range(batch_size):
                    decoder_input[i]=topi[i]

        return total_loss/seq_len, decoder_outputs







if __name__ == '__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder=Encoder(batch_size=2,seq_len=3,embedding_size=4,hidden_size=4,num_layers=2,src_vocab_size=6)
    decoder=Decoder(batch_size=2,embedding_size=4,hidden_size=4,num_layers=2,tar_vocab_size=6)
    src=torch.tensor([[0,2,4],
                      [1,3,5]])
    tar=torch.tensor([[2,4,3,1],
                      [5,0,1,3]])
    criterion=nn.NLLLoss()
    model=Seq2Seq(encoder,decoder,criterion).to(device)
    loss,outputs=model(src,tar)
    print(loss)
    print(outputs)





