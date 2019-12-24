# -*- coding: utf-8 -*-
"""
@author: longshuicui

@date  :  2019/12/23

@modify:

"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


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
                 vocab_size=None):
        """
        编码器初始化 采用双向LSTM网络
        :param seq_len: 序列长度，默认50
        :param embedding_size: 输入向量维度，默认128
        :param hidden_size: 隐藏状态向量维度，默认128
        :param num_layers: lstm层数，默认2
        :param vocab_size: 词典大小
        """
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.hidden_size,
                            bidirectional=True,
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
    def __init__(self):
        super(Decoder, self).__init__()


    def forward(self, x):
        pass








if __name__ == '__main__':
    a = torch.randn([2,3,4])
    mask=torch.zeros([2,1,1,3],dtype=torch.int32)
    res=SelfAttention(4,2)(a,mask,dropout=0.1)
    print(res)

