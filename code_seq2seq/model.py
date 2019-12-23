# -*- coding: utf-8 -*-
"""
@author: longshuicui

@date  :  2019/12/23

@modify:

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


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




if __name__ == '__main__':
    a = torch.tensor([[1, 2, 3, 0],
                      [4, 6, 0, 0]])
    print(a.size())
    encoder = Encoder(seq_len=4, embedding_size=3, hidden_size=3, vocab_size=7)
    outputs, hidden = encoder(a)
