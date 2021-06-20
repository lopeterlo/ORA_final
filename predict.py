from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb
import os
import random
import pickle
import json

from nltk.tokenize import RegexpTokenizer
import torch
import torch.optim as optim
import torch.nn as nn

import generator
import discriminator
import helpers



CUDA = True
VOCAB_SIZE = 25474
MAX_SEQ_LEN = 200
START_LETTER = 0
BATCH_SIZE = 32
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 20
POS_NEG_SAMPLES = 30000

GEN_EMBEDDING_DIM = 300
GEN_HIDDEN_DIM = 300
DIS_EMBEDDING_DIM = 300
DIS_HIDDEN_DIM = 300


testing_samples_path = './data/valid.trc'
testing_id_path = './data/valid_idlist'
gen_path = f'model/ADV_gen_MLEtrain_EMBDIM_300_HIDDENDIM300_VOCAB25474_MAXSEQLEN200_1_06new'


class words_dict():
    def __init__(self):
        self.word_count = {}
        self.id_to_word = {0: '_sos_', 1: '_eos_', 2: '_unk_'}
        self.word_to_id = {'_sos_': 0, '_eos_': 1, '_unk_': 2}
        self.n_words = 3
        self.tokenizer =  RegexpTokenizer(r'\w+')
        self.remain_id = []
        self.max_len = 200
        



# MAIN
def main(dictionary):

    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED) 
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(0)

    with open('data/glove_id_to_emb.pkl', 'rb') as f:
        emb_dict = pickle.load(f)

    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA, emb_dict=emb_dict)

    if CUDA:
        gen = gen.cuda()

    # LOAD GENERATOR 
    gen.load_state_dict(torch.load(gen_path))
    valid_samples = torch.load(testing_samples_path).type(torch.LongTensor)
    valid_id = torch.load(testing_id_path)
    if CUDA:
        gen = gen.cuda()
        valid_samples = valid_samples.cuda()
    with torch.no_grad():
        prediction = ''
        batch = 50
        for idx in range(0, len(valid_samples), batch):
        # for idx, sample in enumerate(valid_samples):
            sample = valid_samples[idx: idx + batch]
            pred = gen.predict(sample, dictionary)
            pred = list(map(' '.join, pred.values()))
            # if idx % 20 == 0:
                # print('pred_summary:', pred[0])
                # print('-' * 50)
            print(f'Validation iteration : {idx}', end = '\r')
            for j, p in enumerate(pred):
                prediction += json.dumps({"id":str(valid_id[idx + j]), "predict": p}) + '\n'

    with open('./prediction/seqgan_0608.json','w') as f:
        f.write(prediction)



if __name__=='__main__':
    dictionary = words_dict()
    with open('./data/train_dict_cut.pkl', 'rb') as f:
        dictionary = pickle.load(f)
    main(dictionary)