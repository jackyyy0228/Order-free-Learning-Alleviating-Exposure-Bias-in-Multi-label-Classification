import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import numpy as np

import data.dataloader as dataloader
import data.utils as utils
import data.dict as dict
from optims import Optim
import lr_scheduler as L
import metrics
from models import encoder_rnn, decoder_rnn, top_k_decoder, seq2vec, decoder_fc
from models.losses import OCDLosses, OrderFreeLosses, CELosses, logit_loss
from models import rescore

import os
import argparse
import time
import json
import collections
import codecs


#config
parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-config', default='config_rnn.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='', type=str,
                    help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-notrain', default=False, action='store_true',
                    help="train or not")
opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

print('#Start:',utils.format_time(time.localtime()))

# checkpoint
if opt.restore: 
    print('loading checkpoint...\n')
    checkpoints = torch.load(opt.restore)
    config = checkpoints['config']
    threshold = checkpoints['threshold']
else:
    threshold = 0.5


if 'train_batch_size' not in config:
    config.train_batch_size = config.batch_size
    config.test_batch_size = config.batch_size
    config.load_emb = False
# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
#use_cuda = True
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)
print(use_cuda)

# data
print('loading data...\n')
start_time = time.time()
datas = torch.load(config.data)  
print('loading time cost: %.3f' % (time.time()-start_time))

trainset, validset, testset = datas['train'], datas['valid'], datas['test']
src_vocab, tgt_vocab = datas['dicts']['src'], datas['dicts']['tgt']
config.src_vocab = src_vocab.size()
config.tgt_vocab = tgt_vocab.size()

## For EOS and SOS
trainset.tgt_vocab_size = tgt_vocab.size() + 2 
validset.tgt_vocab_size = tgt_vocab.size() + 2 
testset.tgt_vocab_size = tgt_vocab.size() + 2 

trainloader = dataloader.get_loader(trainset, batch_size=config.train_batch_size, shuffle=True, 
                                    num_workers=2, max_len = config.max_tgt_len, sos_id = tgt_vocab.size(), 
                                    eos_id = tgt_vocab.size()+1)
validloader = dataloader.get_loader(validset, batch_size=config.test_batch_size, shuffle=False,
                                    num_workers=2, max_len = config.max_tgt_len, sos_id = tgt_vocab.size(), 
                                    eos_id = tgt_vocab.size()+1)
testloader = dataloader.get_loader(testset, batch_size=config.test_batch_size, shuffle=False,
                                    num_workers=2, max_len = config.max_tgt_len, sos_id = tgt_vocab.size(), 
                                    eos_id = tgt_vocab.size()+1)

#metric calculator
train_kinds = set([tuple(x.numpy()) for x in trainset.tgt])
test_kinds = set([tuple(x.numpy()) for x in testset.tgt])
E = metrics.eval_metrics(train_kinds, test_kinds, config.eval_metrics_split_labels)

all_metrics = metrics.eval_metrics().metrics
scores = [[] for metric in all_metrics]
scores = collections.OrderedDict(zip(all_metrics, scores))

if config.eval_metrics_split_labels :
    standard_metric = 'overall_micro_f1'
else:
    standard_metric = 'in_train_micro_f1'

if config.load_emb:
    pretrain_embed = torch.load(config.emb_path)
else:
    pretrain_embed = None

# model
print('building model...\n')
encoder = encoder_rnn.EncoderRNN(src_vocab.size(), config.max_tgt_len, config.hidden_size, 
                                 dropout_p=config.dropout_p, n_layers=config.encoder_n_layers, 
                                 input_dropout_p = config.input_dropout_p, embedding = pretrain_embed,
                                 bidirectional=config.bidirectional, rnn_cell='lstm') 

decoderFC = decoder_fc.DecoderFC(config.hidden_size, tgt_vocab.size() , config.decoder_fc_layers, 
                                config.bidirectional, use_attention = True, dropout_p = config.dropout_p)

model = seq2vec.Seq2vec(encoder, decoderFC) 

if opt.restore:
    model.load_state_dict(checkpoints['model'])

#CUDA
if use_cuda:
    model.cuda()
if len(opt.gpus) > 1:  
    model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)

# optimizer
if opt.restore:
    optim = checkpoints['optim']
else:
    optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                  lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
optim.set_parameters(model.parameters())

if config.schedule:
    scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)

# total number of parameters
param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]

if not os.path.exists(config.log):
    os.makedirs(config.log)

if config.log.endswith('/'):
    log_path = config.log
else:
    log_path = config.log + '/'

if not os.path.exists(log_path):
    os.mkdir(log_path)
logging = utils.logging(log_path+'log.txt') 
if not opt.notrain: 
    logging_train_loss = utils.logging_csv(log_path + 'train_loss.csv',['epoch','updates','log_loss'])
    logging_valid_loss = utils.logging_csv(log_path + 'valid_loss.csv',['epoch','updates','log_loss'])
    logging_metric =  utils.logging_dict_csv(log_path + 'metrics.csv', ['epoch','updates'] + all_metrics)

for k, v in config.items():
    logging("%s:\t%s\n" % (str(k), str(v)))
logging("\n")
logging(repr(model)+"\n\n")  

logging('total number of parameters: %d\n\n' % param_count)

if opt.restore:
    updates = checkpoints['updates']
else:
    updates = 0


# train
def train(epoch):
    global updates
    model.train()
    total_log_loss, total = 0., 0

    if config.schedule:
        scheduler.step()
        print("Decaying learning rate to %g" % scheduler.get_lr()[0])
    
    
    for raw_src, src, src_len, raw_tgt, tgt_vec, tgt_rnn in trainloader:
        if use_cuda:
            src = src.cuda()
            tgt_vec = tgt_vec.cuda()
        model.zero_grad()
        
        log_output = model(src.transpose(0,1), src_len) 
        
        log_loss = logit_loss(log_output, tgt_vec[:,:tgt_vocab.size()]) * config.logistic_weight

        losses = log_loss
        losses.backward()
        optim.step()

        total_log_loss += log_loss.item()
        total += 1
        updates += 1

        if updates % config.print_interval == 0:
            logging(time.strftime("[%H:%M:%S]", time.localtime()))
            logging(" Epoch: %3d, updates: %8d\n" % (epoch, updates))
            logging("Log loss : {:.5f}\n".format(total_log_loss / total))
            logging_train_loss([epoch, updates, total_log_loss / total])
            total_log_loss, total = 0., 0
        
        if updates % config.eval_interval == 0:
            ## TODO different model will have different decoding strategies
            score = eval(epoch, 'valid') 
            logging_metric(score, epoch, updates)
            for metric, value in score.items():
                scores[metric].append(score[metric])
                if metric == standard_metric and score[metric] >= max(scores[metric]):  
                    save_model(log_path+'best_'+metric+'_checkpoint.pt')
            save_model(log_path+'checkpoint.pt')
    
            model.train()

def eval(epoch, eval_type = 'valid'): 
    #decode_type : greedy or beam_search 
    
    total_log_loss, total = 0., 0
    
    y_logistic, y  = [], []
    
    if eval_type == 'valid':
        loader = validloader
        E.is_split_label = False
    else:
        loader = testloader
        E.is_split_label = config.eval_metrics_split_labels

    model.eval()
    for raw_src, src, src_len, raw_tgt, tgt_vec, tgt_rnn in loader:
        if use_cuda:
            src = src.cuda()
            tgt_vec = tgt_vec.cuda()
        
        log_output = model(src.transpose(0,1), src_len) 
        
        log_loss = logit_loss(log_output, tgt_vec[:,:tgt_vocab.size()]) * config.logistic_weight
        
        total_log_loss += log_loss.item()
        total += 1

        y_logistic.append(log_output.detach().cpu().numpy())
        y.append(tgt_vec.cpu().numpy()[:,:tgt_vocab.size()])
    

    logging("{} log loss :{:.5f}\n".format(eval_type, total_log_loss / total))
    if eval_type == 'valid':
        logging_valid_loss([epoch, updates, total_log_loss / total])
    
    def get_score(y, y_score, typ):
        logging("-"*20 + typ + '-'*20 + '\n')
        loss_dict = E.compute(y, y_score)
        logging(E.logging(loss_dict))
        return loss_dict
    
    y = np.vstack(y)
    y_score = np.vstack(y_logistic)
    np.save(os.path.join(config.log,'y_score.npy'),y_score)
    np.save(os.path.join(config.log,'y.npy'),y)

    E.set_thres(0.5)
    get_score(y, y_score, 'Logistic')
    
    if eval_type == 'valid': 
        global threshold
        _,threshold = E.find_best_thres(y, y_score) 
    E.set_thres(threshold)
    loss_d = get_score(y, y_score, 'Logistic')
    
    logging('-'*50+'\n')
    
    return loss_d
def test(load_checkpoint = False):
    if load_checkpoint:
        checkpoints = torch.load(log_path+'best_{}_checkpoint.pt'.format(standard_metric))
        model.load_state_dict(checkpoints['model'])
        threshold = checkpoints['threshold']
        logging("Best model was selected at {} updates.\n".format(checkpoints['updates']))
    loss_dict = eval(0, 'test')
    return loss_dict

def save_model(path):
    global updates, threshold
    model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates,
        'threshold': threshold}
    torch.save(checkpoints, path)

def main():
    end_epoch = 0
    if opt.notrain:
        test()
        exit()
    for i in range(1, config.epoch+1):
        end_epoch = i
        try:
            train(i)
        except KeyboardInterrupt:
            logging('Interupt\n')
            break
    idx = np.argmax(scores[standard_metric])
    best_epoch = idx+1
    
    logging("Summary (validation):\n")
    for metric in all_metrics:
        logging("{}:{:.3f}\n".format(metric,scores[metric][idx]))
    logging("\nPerformance on test set:\n") 
    test_d = test(True)
    
if __name__ == '__main__':
    main()
