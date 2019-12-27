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
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            temp=[0]*(src_max_len-len(pair[0]))
            temp.extend(pair[0])
            pair[0]=temp
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
    criterion=nn.CrossEntropyLoss()
    model=Seq2Seq(encoder=encoder,decoder=decoder,criterion=criterion).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    logging.info('='*10+'Starting training Model...')
    for iter in range(1,args.iters+1):
        for i, batch_data in enumerate(generator):
            src_data=torch.tensor(batch_data[:,0].tolist()).to(device)
            tar_data=torch.tensor(batch_data[:,1].tolist()).to(device)

            #TODO 模型的损失一直都是0，效果不好，增加注意力机制试一试
            loss,outputs=model(src_data,tar_data)
            logging.info('=' * 10 + "【当前迭代%d，批次%d】, 当前损失:%.4f" % (iter,i + 1, loss))
            for param_group in opt.param_groups:
                if iter<=2:
                    param_group['lr']=0.1
                else:
                    param_group['lr']=args.lr

            opt.zero_grad()
            loss.backward()
            opt.step()

    torch.save({'model_state_dict':model.state_dict(),'src_lang':src_lang,'tar_lang':tar_lang},args.save_path)


def test(args):
    logging.info('=' * 10 + '加载模型')
    params=torch.load(args.save_path)
    src_lang,tar_lang=params['src_lang'],params['tar_lang']
    encoder = Encoder(batch_size=args.batch_size,
                      seq_len=args.src_max_len,
                      embedding_size=args.embedding_size,
                      hidden_size=args.hidden_dim,
                      num_layers=args.num_layers,
                      src_vocab_size=src_lang.n_words)
    decoder = Decoder(batch_size=args.batch_size,
                      embedding_size=args.embedding_size,
                      hidden_size=args.hidden_dim,
                      num_layers=args.num_layers,
                      tar_vocab_size=tar_lang.n_words)
    model = Seq2Seq(encoder=encoder, decoder=decoder, flag=False).to(device)
    model.load_state_dict(params['model_state_dict'])
    model.eval()
    print('ok')




def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-batch_size',type=int,default=128)
    parser.add_argument('-src_max_len',type=int,default=50)
    parser.add_argument('-tar_max_len',type=int,default=61)
    parser.add_argument('-embedding_size',type=int,default=128)
    parser.add_argument('-hidden_dim',type=int,default=128)
    parser.add_argument('-num_layers',type=int,default=2)
    parser.add_argument('-lr',type=float,default=0.001)
    parser.add_argument('-iters',type=int,default=10)
    parser.add_argument('-save_path',type=str,default='./nmt_seq2seq.ckpt')
    parser.add_argument('-teacher_forcing',type=bool,default=True)
    parser.add_argument('-beam_search',type=bool,default=False)

    args=parser.parse_args()

    train(args)
    test(args)


if __name__ == '__main__':
    main()