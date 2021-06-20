import collections
import random

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init

from torch.autograd import Variable


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, bidirectional = False):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        #hidden = [batch size, 1, dec hid dim]
        #encoder_outputs = [batch size, src len, hidden_size]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        #repeat decoder hidden state src_len times

        hidden = hidden.repeat(1, src_len, 1)  #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), -1))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        return F.softmax(attention, dim=1)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers =1, bidirectional=False, dropout = 0):
        super(EncoderRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                     dropout= dropout, bidirectional=bidirectional, batch_first = True)

        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights()
        self.bidirectional = bidirectional
        self.drop = nn.Dropout(dropout)
        
        self.init_weights()

    def init_weights(self):
        for name, p in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0)


    def forward(self, x):
        out, (hn, cn) = self.lstm(x)# out: tensor of shape (batch_size, seq_length, hidden_size)
        hn = self.tanh(self.drop(self.l1(hn)))
        out = self.tanh(self.drop(self.l1(out)))
        return out, hn

    def init_weights(self):
        for name, p in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0)

class Generator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, oracle_init=False, emb_dict=None):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.encoder = EncoderRNN(embedding_dim, hidden_dim)
        self.gpu = gpu

        # self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(2*embedding_dim, hidden_dim)
        self.lstm2out = nn.Linear(2*hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.emb_dict = emb_dict

        # initialise oracle network with N(0,1)
        # otherwise variance of initialisation is very small => high NLL for data sampled from the same model
        if oracle_init:
            for p in self.parameters():
                init.normal(p, 0, 1)

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, inp, h, c):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        """
        # input dim                                             # batch_size
        out, (hidden, cn) = self.lstm(inp, (h,c))                     # 1 x batch_size x hidden_dim (out)
      # out = F.log_softmax(out, dim=1)
        return out, hidden, cn

    def top_index(self, out):
        out = self.lstm2out(out)
        _, idx = out.topk(1)
        return out, idx

    def attn_pred(self, x, enc_outputs): ## x is h, hidden state
        x = x.permute(1, 0, 2) # torch.Size([B, 1, 300])
        weight = self.attention(x, enc_outputs)
        weight = weight.unsqueeze(1)
        context = torch.bmm(weight, enc_outputs) # torch.Size([B, 1, 300])
        concat = torch.cat((context, x), -1) # torch.Size([B, 1, 600])
        concat = concat.permute(1,0,2) # torch.Size([1, 16, 600])
        out = self.lstm2out(concat)
        val, idx = out.max(-1)
        return out, idx

    def sample(self, inp, word_dict):
        """
        Samples the network and returns num_samples samples of length max_seq_len.

        Outputs: samples, hidden
            - samples: num_samples x max_seq_length (a sampled sequence in each row)
        """
        batch_size, seq_len = inp.size()

        # loss_fn = nn.NLLLoss()

        emb = torch.tensor([[self.emb_dict.get(x.item()) for x in sam] for sam in inp]).float().cuda() 
        enc_out, h = self.encoder(emb) ## out : batch_size, seq_length, hidden_size
        # print(enc_out.shape, h.shape) # torch.Size([128, 200, 300]) torch.Size([1, 128, 300])
        context = h
        cn = Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda()


        g_inp = torch.zeros(batch_size).cuda()
       
        # h = self.init_hidden(batch_size)
        words = g_inp
        word_emb = torch.tensor([[self.emb_dict.get(x.item()) for x in words]]).float().cuda()
        inputs = torch.cat((context, word_emb), 2)

        loss = 0

        prediction = collections.defaultdict(list)
        prediction_idx = collections.defaultdict(list)
        for j in range(batch_size):
            prediction[j].append('_sos_')
            prediction_idx[j].append(0)

        # h = self.init_hidden(batch_size)
        for i in range(seq_len):
            # print(g_inp.shape, h.shape, cn.shape) #torch.Size([1, 50]) torch.Size([1, 50, 300]) torch.Size([1, 50, 300])
            out, h, cn = self.forward(inputs, h, cn)
            
            ## abstractive
            # combined = torch.cat((context, h), -1)
            # out, idx = self.top_index(combined)
            ## attention
            out, idx = self.attn_pred(h, enc_out)
            # print(out.shape, idx.shape) # torch.Size([1, 128, 25474]) torch.Size([1, 128])
            for j in range(batch_size):
                prediction[j].append(word_dict.id_to_word[idx.tolist()[0][j]])
                prediction_idx[j].append(idx.tolist()[0][j])
            word_emb = torch.tensor([[self.emb_dict.get(prediction_idx[k][i]) for k in range(batch_size)]]).float().cuda()
            inputs = torch.cat((h, word_emb), -1)
            if i >= 30:
                break
        ans = torch.tensor(list(prediction_idx.values()))
        batch, length = ans.size()
        end = torch.ones(batch, seq_len - length).long()
        concate = torch.cat((ans,end), axis = 1).cuda()
        return concate
        

    def batchNLLLoss(self, inp, target):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - inp: batch_size x seq_len , [[1,2,3], []]
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """
        batch_size, seq_len = target.size()

        # loss_fn = nn.NLLLoss()

        emb = torch.tensor([[self.emb_dict.get(x.item()) for x in sam] for sam in inp]).float().cuda() # batch_size x seq_len x emb
        enc_out, h = self.encoder(emb) ## out : batch_size, seq_length, hidden_size
        # print(enc_out.shape, h.shape) # torch.Size([128, 200, 300]) torch.Size([1, 128, 300])
        context = h
        cn = Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda()

        loss_fn = nn.CrossEntropyLoss()
        
        g_inp = target.clone()
        ones = torch.ones(batch_size, 1).long().cuda()
        target = torch.cat((target[:, 1:], ones), axis = 1)

        g_inp = g_inp.permute(1, 0)       # seq_len x batch_size
        target = target.permute(1, 0)     # seq_len x batch_size
        
        # h = self.init_hidden(batch_size)
        loss = 0
        words = g_inp[0]

        word_emb = torch.tensor([[self.emb_dict.get(x.item()) for x in words]]).float().cuda()
        inputs = torch.cat((context, word_emb), 2)

        # print(words.shape, word_emb.shape, inputs.shape) # torch.Size([128]) torch.Size([1, 128, 300]) torch.Size([1, 128, 600])

        for i in range(seq_len):
            # print(g_inp.shape, h.shape, cn.shape)  torch.Size([200, 128]) torch.Size([1, 128, 300]) torch.Size([1, 128, 300])
            # print(inputs.shape ,h.shape)
            out, h, cn = self.forward(inputs, h, cn)
            ## abstractive
            # combined = torch.cat((context, h), -1)
            # out, idx = self.top_index(combined)
            ## attention
            out, idx = self.attn_pred(h, enc_out)
            # print(out.shape, idx.shape) # torch.Size([1, 128, 25474]) torch.Size([1, 128])

            loss += loss_fn(out.squeeze(0), target[i])
            teacher_forcing = 0.5
            use_TF = True if random.random() < teacher_forcing else False
            if use_TF:
                words = target[i] # [batch_size]
            else:
                words = idx[0] # [batch_size]

            word_emb = torch.tensor([[self.emb_dict.get(x.item()) for x in words]]).float().cuda()
            inputs = torch.cat((h, word_emb), -1)
            if i >= 30:
                break
        # print(loss)
        return loss     # per batch

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """
        ori_target = target.clone()
        batch_size, seq_len = target.size()

        # loss_fn = nn.NLLLoss()

        emb = torch.tensor([[self.emb_dict.get(x.item()) for x in sam] for sam in inp]).float().cuda() # batch_size x seq_len x emb
        enc_out, h = self.encoder(emb) ## out : batch_size, seq_length, hidden_size
        # print(enc_out.shape, h.shape) # torch.Size([128, 200, 300]) torch.Size([1, 128, 300])
        context = h
        cn = Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda()

        loss_fn = nn.CrossEntropyLoss()
        
        g_inp = target.clone()
        ones = torch.ones(batch_size, 1).long().cuda()
        target = torch.cat((target[:, 1:], ones), axis = 1)

        g_inp = g_inp.permute(1, 0)       # seq_len x batch_size
        target = target.permute(1, 0)     # seq_len x batch_size
        
        # h = self.init_hidden(batch_size)
        loss = 0
        words = g_inp[0]

        word_emb = torch.tensor([[self.emb_dict.get(x.item()) for x in words]]).float().cuda()
        inputs = torch.cat((context, word_emb), 2)
        
        # h = self.init_hidden(batch_size)
        loss = 0
        for i in range(seq_len):
            # print(g_inp.shape, h.shape, cn.shape)  torch.Size([200, 128]) torch.Size([1, 128, 300]) torch.Size([1, 128, 300])
            # print(inputs.shape ,h.shape)
            out, h, cn = self.forward(inputs, h, cn)
            ## abstractive
            # combined = torch.cat((context, h), -1)
            # out, idx = self.top_index(combined)
            ## attention
            out, idx = self.attn_pred(h, enc_out)
            out = self.softmax(out)
            # print(out.shape, idx.shape) # torch.Size([1, 128, 25474]) torch.Size([1, 128])
            for j in range(batch_size):
                loss += -out[0][j][ori_target.data[j][i]]*reward[j]  * (1 - i / seq_len)

            teacher_forcing = 0
            use_TF = True if random.random() < teacher_forcing else False
            if use_TF:
                words = target[i] # [batch_size]
            else:
                words = idx[0] # [batch_size]

            word_emb = torch.tensor([[self.emb_dict.get(x.item()) for x in words]]).float().cuda()
            inputs = torch.cat((h, word_emb), -1)
            if i >= 30:
                break

        
        return loss/batch_size

    def predict(self, inp, word_dict):


        batch_size, seq_len = inp.size()

        # loss_fn = nn.NLLLoss()

        emb = torch.tensor([[self.emb_dict.get(x.item()) for x in sam] for sam in inp]).float().cuda() 
        enc_out, h = self.encoder(emb) ## out : batch_size, seq_length, hidden_size
        # print(enc_out.shape, h.shape) # torch.Size([128, 200, 300]) torch.Size([1, 128, 300])
        context = h
        cn = Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda()


        g_inp = torch.zeros(batch_size).cuda()
       
        # h = self.init_hidden(batch_size)
        loss = 0
        words = g_inp
        word_emb = torch.tensor([[self.emb_dict.get(x.item()) for x in words]]).float().cuda()
        inputs = torch.cat((context, word_emb), 2)

        loss = 0
        prediction = collections.defaultdict(list)
        prediction_idx = collections.defaultdict(list)
        # h = self.init_hidden(batch_size)
        for i in range(seq_len):
            # print(g_inp.shape, h.shape, cn.shape) #torch.Size([1, 50]) torch.Size([1, 50, 300]) torch.Size([1, 50, 300])
            out, h, cn = self.forward(inputs, h, cn)
            ## abstractive
            # combined = torch.cat((context, h), -1)
            # out, idx = self.top_index(combined)
            ## attention
            out, idx = self.attn_pred(h, enc_out)
            # print(out.shape, idx.shape) # torch.Size([1, 128, 25474]) torch.Size([1, 128])
            for j in range(batch_size):
                prediction[j].append(word_dict.id_to_word[idx.tolist()[0][j]])
                prediction_idx[j].append(idx.tolist()[0][j])
            word_emb = torch.tensor([[self.emb_dict.get(prediction_idx[k][i]) for k in range(batch_size)]]).float().cuda()
            inputs = torch.cat((h, word_emb), -1)
            if i >= 30:
                break
        return prediction