from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb
import os
import random

import pickle
from nltk.tokenize import RegexpTokenizer
import torch
import torch.optim as optim
import torch.nn as nn

import generator
import discriminator
import helpers

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

CUDA = True
VOCAB_SIZE = 25474
MAX_SEQ_LEN = 200
START_LETTER = 0
BATCH_SIZE = 64
MLE_TRAIN_EPOCHS = 100
ADV_TRAIN_EPOCHS = 20
POS_NEG_SAMPLES = 10000

GEN_EMBEDDING_DIM = 300
GEN_HIDDEN_DIM = 300
DIS_EMBEDDING_DIM = 300
DIS_HIDDEN_DIM = 300



train_samples_path = './data/train.trc'
summary_samples_path = './data/train_summary.trc'
val_samples_path = './data/valid.trc'
val_summary_samples_path = './data/valid_summary.trc'

# oracle_state_dict_path = './oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
pretrained_gen_path = f'./model/final_0609_seqgan_0.1897_@30'
pretrained_dis_path = f'model/dis_pretrain_EMBDIM_{DIS_EMBEDDING_DIM}_HIDDENDIM{DIS_HIDDEN_DIM}_VOCAB{VOCAB_SIZE }_MAXSEQLEN{MAX_SEQ_LEN}'
adv_gen = f'model/ADV_gen_MLEtrain_EMBDIM_{GEN_EMBEDDING_DIM}_HIDDENDIM{GEN_HIDDEN_DIM}_VOCAB{VOCAB_SIZE }_MAXSEQLEN{MAX_SEQ_LEN}'
adv_dis = f'model/ADV_dis_pretrain_EMBDIM_{DIS_EMBEDDING_DIM}_HIDDENDIM{DIS_HIDDEN_DIM}_VOCAB{VOCAB_SIZE }_MAXSEQLEN{MAX_SEQ_LEN}'


class words_dict():
    def __init__(self):
        self.word_count = {}
        self.id_to_word = {0: '_sos_', 1: '_eos_', 2: '_unk_'}
        self.word_to_id = {'_sos_': 0, '_eos_': 1, '_unk_': 2}
        self.n_words = 3
        self.tokenizer =  RegexpTokenizer(r'\w+')
        self.remain_id = []
        self.max_len = 200
        


def train_generator_MLE(gen, gen_opt, oracle, real_data_samples, summary_samples, val_samples, val_summary, dictionary, epochs, sample_num =POS_NEG_SAMPLES):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        print('epoch %d / %d : ' % (epoch + 1,MLE_TRAIN_EPOCHS), end='\n')
        sys.stdout.flush()
        total_loss = 0
        # idx = random.randint(0,6)
        # for i in range(idx * POS_NEG_SAMPLES, (idx + 1) * POS_NEG_SAMPLES, BATCH_SIZE):
        index = random.choices(list(range(len(real_data_samples))), k =sample_num)
        sample_real_data_samples, sample_summary_samples = real_data_samples[index].float().clone(), summary_samples[index].float().clone()
        for i in range(0, sample_num, BATCH_SIZE):
            s, e = i, i + BATCH_SIZE
            inp, target = sample_real_data_samples[s:e], sample_summary_samples[s:e]
            inp, target = helpers.prepare_generator_batch(inp, target, start_letter=START_LETTER,
                                                          gpu=CUDA)
            # print(inp.shape, target.shape)
            
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()
            print(f'Iteration: {i} / {sample_num}, Loss {loss.item()}', end = '\r')
            # if (i / BATCH_SIZE) % ceil(
            #                 ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
            #     print('.', end='')
            #     sys.stdout.flush()
        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(sample_num / float(BATCH_SIZE)) / MAX_SEQ_LEN

        # sample from generator and compute oracle NLL
        oracle_loss = 0

        print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))
        print('-' * 50)
        # val_sentence = ''
        # for id in val_samples[epoch]:
        #     val_sentence += dictionary.id_to_word[id.item()] + ' '
        # print('val_sample:', val_sentence)

        val_sum = ''
        for id in val_summary[epoch]:
            val_sum += dictionary.id_to_word[id.item()] + ' '
        print('val_summary:', val_sum)

        pred = gen.predict(val_samples[0].unsqueeze(0), dictionary)
        pred = ' '.join(pred[0])
        print('pred_summary:', pred)
        print('-' * 50)

        if epoch != 0 and epoch % 10 == 0:
            torch.save(gen.state_dict(), pretrained_gen_path + f'_{epoch}_0609')

def train_generator_PG(gen, gen_opt, oracle, dis, num_batches, train_samples, summary_samples, dictionary):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """
    smaple_num = 10000

    index = random.choices(list(range(len(train_samples))), k = smaple_num)
    sample_real_data_samples, sample_summary_samples = train_samples[index].float().clone(), summary_samples[index].float().clone()
    for i in range(0, smaple_num, num_batches):
        s, e = i, i + BATCH_SIZE
        inp, target = sample_real_data_samples[s:e], sample_summary_samples[s:e]
        inp, target = helpers.prepare_generator_batch(inp, target, start_letter=START_LETTER,
                                                      gpu=CUDA)
        # print(inp.shape, target.shape)
        rewards = dis.batchClassify(target)
        
        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()


        print(f'Iteration: {i} / {smaple_num}, Loss {pg_loss.item()}', end = '\r')


    # sample from generator and compute oracle NLL
    # oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN, summary_samples, 
    #                                                start_letter=START_LETTER, gpu=CUDA)

    # print(' oracle_sample_NLL = %.4f' % oracle_loss)


def train_discriminator(discriminator, dis_opt, train_samples, summary_samples,  generator, oracle, dictionary, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using oracle and generator)
    # pos_val = oracle.sample(100) oracle not used
    val_smaple_num = 100
    idx = random.choices([i for i in range(len(summary_samples)- val_smaple_num)], k = 1)
    pos_val = summary_samples[idx[0]:idx[0] + val_smaple_num]
    neg_val = generator.sample(train_samples[idx[0]:idx[0] + val_smaple_num], dictionary)
    val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    smaple_num = 5000
    for d_step in range(d_steps):
        idx = random.choices([i for i in range(len(summary_samples) - smaple_num)], k = 1)
        batch_summary = summary_samples[idx[0]:idx[0] + smaple_num]
        s = helpers.batchwise_sample(generator, train_samples, idx, dictionary, smaple_num, BATCH_SIZE)
        dis_inp, dis_target = helpers.prepare_discriminator_data(batch_summary, s)
        # print(dis_inp.device, dis_target.device)

        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i in range(0, 2 * smaple_num, BATCH_SIZE):
                inp, target = dis_inp[i:i + BATCH_SIZE], dis_target[i:i + BATCH_SIZE]

                if CUDA :
                    inp = inp.cuda()
                    target = target.cuda()

                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                # if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                #         BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                #     print('.', end='')
                #     sys.stdout.flush()
                print(f'Iteration: {i} / {smaple_num *2}, Loss {loss.item()}', end = '\r')
            total_loss /= ceil(2 * smaple_num / float(BATCH_SIZE))
            total_acc /= float(2 * smaple_num)
            print('\nstart validation')
            val_pred = discriminator.batchClassify(val_inp)
            print(torch.sum((val_pred>0.5)==(val_target>0.5)).data.item())
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/200))

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

    oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA, oracle_init = True,  emb_dict= emb_dict)
    # oracle.load_state_dict(torch.load(oracle_state_dict_path))
    train_samples = torch.load(train_samples_path).type(torch.LongTensor)
    summary_samples = torch.load(summary_samples_path).type(torch.LongTensor)
    valid_samples = torch.load(val_samples_path).type(torch.LongTensor)
    valid_summary = torch.load(val_summary_samples_path).type(torch.LongTensor)
    # a new oracle can be generated by passing oracle_init=True in the generator constructor
    # samples for the new oracle can be generated using helpers.batchwise_sample()

    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA, emb_dict= emb_dict)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA, emb_dict= emb_dict)

    if CUDA:
        oracle = oracle.cuda()
        gen = gen.cuda()
        dis = dis.cuda()
        # train_samples = train_samples.cuda()
        summary_samples = summary_samples.cuda()
        # valid_samples = valid_samples.cuda()
        valid_summary = valid_summary.cuda()

    # GENERATOR MLE TRAINING
    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-3)

    train_generator_MLE(gen, gen_optimizer, oracle, train_samples, summary_samples, valid_samples, valid_summary,dictionary,  MLE_TRAIN_EPOCHS)
    torch.save(gen.state_dict(), pretrained_gen_path)

    # gen.load_state_dict(torch.load(pretrained_gen_path))

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adam(dis.parameters(), lr = 1e-3)
    train_discriminator(dis, dis_optimizer, train_samples, summary_samples,  gen, oracle, dictionary, 1, 1)
    torch.save(dis.state_dict(), pretrained_dis_path)

    # dis.load_state_dict(torch.load(pretrained_dis_path))

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')
    # oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN, summary_samples,  
    #                                            start_letter=START_LETTER, gpu=CUDA)
    # print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)
    gen.train()
    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        train_generator_PG(gen, gen_optimizer, oracle, dis, BATCH_SIZE, train_samples, summary_samples, dictionary)
        # train_generator_MLE(gen, gen_optimizer, oracle, train_samples, summary_samples, valid_samples, valid_summary, dictionary, 1)
        if epoch != 0 and epoch % 3 == 0:
            # TRAIN DISCRIMINATOR
            print('\nAdversarial Training Discriminator : ')
            train_discriminator(dis, dis_optimizer, train_samples, summary_samples,  gen, oracle, dictionary, 1, 1)
        
        torch.save(gen.state_dict(), adv_gen + f'_{epoch}_06new')
        # torch.save(dis.state_dict(), adv_dis + f'_{epoch}')


if __name__=='__main__':
    dictionary = words_dict()
    with open('./data/train_dict_cut.pkl', 'rb') as f:
        dictionary = pickle.load(f)
    main(dictionary)