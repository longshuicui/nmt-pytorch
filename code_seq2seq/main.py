# -*- coding: utf-8 -*-
"""
@author: longshuicui

@date  :  2019/12/24

@modify:

"""
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn

from data_process import load_data, text2tensor
from model import Encoder,Decoder,Seq2Seq

logging.basicConfig(level=logging.INFO,format="%(asctime)s %(name)s %(levelname)s %(message)s")

def gen_batch_data(data=None,batch_size=64,src_max_len=50,tar_max_len=60,shuffle=True):
    """
    生成batch训练数据
    :param data: 数据 格式[src,tar]
    :param batch_size:batch 大小
    :param shuffle:是否打乱数据
    :return:
    """
    logging.info('='*10+'生成 batch 数据')
    num=len(data)
    data=np.array(data,dtype=np.object)
    if shuffle:
        shuffle_id=np.random.permutation(num)
        data=data[shuffle_id]

    #数据填充，不足的填充0
    batch_data = []
    for pair in data:
        if len(pair[0])<src_max_len:
            pair[0].extend([0]*(src_max_len-len(pair[0])))
        if len(pair[1])<tar_max_len:
            pair[1].extend([0]*(tar_max_len-len(pair[1])))
    for i in range(num):
        batch_data.append(data[i])
        if len(batch_data)==batch_size or i==num-1:
            yield np.array(batch_data)
            batch_data=[]

def train(args):
    src_lang, tar_lang, pairs=load_data()
    vectors=text2tensor(src_lang,tar_lang,pairs)
    generator=gen_batch_data(vectors,args.batch_size,args.src_max_len,args.tar_max_len)

    logging.info('='*10+'模型初始化')
    encoder=Encoder(batch_size=args.batch_size,
                    seq_len=args.src_max_len,
                    embedding_size=args.embedding_size,
                    hidden_size=args.hidden_dim,
                    num_layers=args.num_layers,
                    src_vocab_size=src_lang.n_words)
    decoder=Decoder(batch_size=args.batch_size,
                    embedding_size=args.embedding_size,
                    hidden_size=args.hidden_dim,
                    num_layers=args.num_layers,
                    tar_vocab_size=tar_lang.n_words)
    criterion=nn.NLLLoss()
    model=Seq2Seq(encoder=encoder,decoder=decoder,criterion=criterion)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    for i, batch_data in enumerate(generator):
        src_data=torch.tensor(batch_data[:,0].tolist())
        tar_data=torch.tensor(batch_data[:,1].tolist())
        loss,outputs=model(src_data,tar_data)
        logging.info('=' * 10 + "当前迭代%d, 当前损失:%.4f" % (i + 1, loss))
        print(outputs)
        opt.zero_grad()
        loss.backward()
        opt.step()





def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-batch_size',type=int,default=64)
    parser.add_argument('-src_max_len',type=int,default=50)
    parser.add_argument('-tar_max_len',type=int,default=70)
    parser.add_argument('-embedding_size',type=int,default=128)
    parser.add_argument('-hidden_dim',type=int,default=256)
    parser.add_argument('-num_layers',type=int,default=2)
    parser.add_argument('-lr',type=float,default=0.001)
    parser.add_argument('-teacher_forcing',type=bool,default=True)
    parser.add_argument('-beam_search',type=bool,default=False)

    args=parser.parse_args()

    train(args)



if __name__ == '__main__':
    main()