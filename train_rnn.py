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
from models import encoder_rnn, decoder_rnn, top_k_decoder, seq2seq, decoder_fc
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

parser.add_argument('-config', default='config/config_aapd.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='', type=str,
                    help="restore checkpoint")
parser.add_argument('-seed', type=int, default=7122,
                    help="Random seed")
parser.add_argument('-notrain', default=False, action='store_true',
                    help="train or not")
opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

print('#Start:',utils.format_time(time.localtime()))

threshold = 0.5
# checkpoint
if opt.restore: 
    print('loading checkpoint...\n')
    checkpoints = torch.load(opt.restore)
    config = checkpoints['config']
    threshold = checkpoints['threshold']
if 'train_batch_size' not in config:
    config.train_batch_size = config.batch_size
    config.test_batch_size = config.batch_size
    config.load_emb = False
config.test_batch_size= 32

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

standard_metric = 'in_train_micro_f1'

# pretrain embedding
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


decoder = decoder_rnn.DecoderRNN(tgt_vocab.size() + 2, config.max_tgt_len, config.hidden_size*2,
                                tgt_vocab.size() , tgt_vocab.size() + 1, config.loss_type, rnn_cell = 'lstm', 
                                 n_layers = config.decoder_n_layers, use_attention = True, 
                                 bidirectional = config.bidirectional, dropout_p = config.dropout_p,
                                 input_dropout_p = config.input_dropout_p,
                                 sampling_type=config.decoder_sampling_type, add_mask = config.add_mask)
decoderFC = decoder_fc.DecoderFC(config.hidden_size, tgt_vocab.size() , config.decoder_fc_layers, 
                                config.bidirectional, use_attention = True, dropout_p = config.dropout_p)

model = seq2seq.Seq2seq(encoder, decoder, decoderFC) 

## Loss
if config.loss_type.lower() == 'ocd':
    Loss = OCDLosses(tgt_vocab.size() + 1, config.OCD_temperature_start,
                        config.OCD_temperature_end, config.OCD_final_hard_epoch)
elif config.loss_type.lower() == 'vanilla':
    Loss = CELosses(tgt_vocab.size() + 1)

elif config.loss_type == 'order_free':
    Loss = OrderFreeLosses(tgt_vocab.size() + 1)


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

#logging
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
    logging_train_loss = utils.logging_csv(log_path + 'train_loss.csv',['epoch','updates','log_loss','rnn_loss'])
    logging_valid_loss = utils.logging_csv(log_path + 'valid_loss.csv',['epoch','updates','log_loss','rnn_loss'])
    logging_metric_joint = utils.logging_dict_csv(log_path + 'metrics_joint.csv', ['epoch','updates'] + all_metrics)
    logging_metric =  utils.logging_dict_csv(log_path + 'metrics.csv', ['epoch','updates'] + all_metrics)
    logging_metric_logistic =  utils.logging_dict_csv(log_path + 'metrics_log.csv', ['epoch','updates'] + all_metrics)
    
for k, v in config.items():
    logging("%s:\t%s\n" % (str(k), str(v)))
logging("\n")
logging(repr(model)+"\n\n")  

logging('total number of parameters: %d\n\n' % param_count)

if opt.restore:
    updates = checkpoints['updates']
else:
    updates = 0

#threshold for binary classifier
threshold = 0.5

# train
def train(epoch):
    global updates
    model.train()
    model.decoder.set_sampling_type(config.decoder_sampling_type)
    total_log_loss, total_rnn_loss, total = 0., 0., 0
    #optim.updateLearningRate(None, epoch)
    if config.schedule:
        scheduler.step()
        print("Decaying learning rate to %g" % scheduler.get_lr()[0])
    
    ## Update Teacher Forcing ratio
    if config.loss_type.lower() == 'vanilla' or config.loss_type.lower() == 'order_free':
        if epoch > config.teacher_forcing_final_epoch:
            teacher_forcing_ratio = config.teacher_forcing_ratio_end 
        else:
            teacher_forcing_ratio = config.teacher_forcing_ratio_start + (config.teacher_forcing_ratio_end - config.teacher_forcing_ratio_start) / config.teacher_forcing_final_epoch * (epoch-1)
        logging("Teacher forcing ratio: " + str(teacher_forcing_ratio) + '\n')
    else:
        teacher_forcing_ratio = 0
    # Update Temperature
    if config.loss_type.lower() == 'ocd':
        Loss.update_temperature(epoch)
    
    for raw_src, src, src_len, raw_tgt, tgt_vec, tgt_rnn in trainloader:
        if use_cuda:
            src = src.cuda()
            tgt_vec = tgt_vec.cuda()
            tgt_rnn = tgt_rnn.cuda()
        model.zero_grad()
        
        target_variable = None
        candidates = tgt_vec.clone()
        label_sets = tgt_vec.clone()
        
        if config.loss_type.lower() == 'vanilla':
            target_variable = tgt_rnn
            label_sets = tgt_rnn
        
        decoder_outputs, decoder_hidden, ret_dict, log_output = model(src.transpose(0,1), src_len, 
                                                                target_variable = target_variable, 
                                                                candidates = candidates,
                                                                teacher_forcing_ratio=teacher_forcing_ratio)

        rnn_loss = Loss(decoder_outputs, ret_dict['sequence'], label_sets)  
        
        log_loss = logit_loss(log_output, tgt_vec[:,:tgt_vocab.size()]) * config.logistic_weight

        losses = rnn_loss + log_loss
        losses.backward()
        optim.step()

        total_log_loss += log_loss.item()
        total_rnn_loss += rnn_loss.item()
        total += 1
        updates += 1

        if updates % config.print_interval == 0:
            logging(time.strftime("[%H:%M:%S]", time.localtime()))
            logging(" Epoch: %3d, updates: %8d\n" % (epoch, updates))
            logging("RNN loss : {:.5f} \nLog loss : {:.5f}\n".format(total_rnn_loss / total, total_log_loss / total))
            logging_train_loss([epoch,updates,total_log_loss / total, total_rnn_loss / total])
            total_log_loss, total_rnn_loss, total = 0., 0., 0
        
        if updates % config.eval_interval == 0:
            ## TODO different model will have different decoding strategies
            score_rnn, score_logistic = eval(epoch, 'valid', 'greedy', False)
            logging_metric(score_rnn, epoch, updates)
            logging_metric_logistic(score_logistic, epoch, updates)
            if config.logistic_weight > 0:
                score_joint,_ =  eval(epoch, 'valid', 'beam_search', True)
                logging_metric_joint(score_joint , epoch, updates)
            score = score_rnn
            #eval(epoch, 'test', 'greedy', True)
            for metric, value in score.items():
                scores[metric].append(score[metric])
                if metric == standard_metric and score[metric] >= max(scores[metric]):  
                    save_model(log_path+'best_'+metric+'_checkpoint.pt')
            save_model(log_path+'checkpoint.pt')
    
            model.train()
            model.decoder.set_sampling_type(config.decoder_sampling_type)

def eval(epoch, eval_type = 'valid', decode_type = 'greedy', 
         logistic_joint_decoding = False):
    #decode_type : greedy or beam_search 
    
    total_rnn_loss, total_log_loss, total = 0.,0., 0.
    
    y_logistic, y_rnn, y_rescore, y  = [], [], [], []
    
    if eval_type == 'valid':
        loader = validloader
        E.is_split_label = False
    else:
        loader = testloader
        E.is_split_label = config.eval_metrics_split_labels

    model.decoder.set_sampling_type('max') 
    if decode_type == 'beam_search':
        topk_decoder = top_k_decoder.TopKDecoder(model.decoder, config.beam_size, config.beam_score_type)
        eval_model =  seq2seq.Seq2seq(encoder, topk_decoder, decoderFC) 
    elif decode_type == 'greedy':
        eval_model = model
    
    eval_model.eval()
    for raw_src, src, src_len, raw_tgt, tgt_vec, tgt_rnn in loader:
        if use_cuda:
            src = src.cuda()
            tgt_vec = tgt_vec.cuda()
            tgt_rnn = tgt_rnn.cuda()
        
        decoder_outputs, decoder_hidden, ret_dict, log_output = eval_model(src.transpose(0,1), src_len, 
                                                                           logistic_joint_decoding = logistic_joint_decoding)
        
        if config.loss_type.lower() == 'vanilla':
            label_sets = tgt_rnn
        else:
            label_sets = tgt_vec.clone()

        rnn_loss = Loss(decoder_outputs, ret_dict['sequence'], label_sets)
        log_loss = logit_loss(log_output, tgt_vec[:,:tgt_vocab.size()]) * config.logistic_weight
        
        total_log_loss += log_loss.item()
        total_rnn_loss += rnn_loss.item()
        total += 1

        y_vec = E.idx2vec(ret_dict['sequence'], tgt_vocab.size(), tgt_vocab.size()+1, True)
        y_rnn.append(y_vec)
        y_logistic.append(log_output.detach().cpu().numpy())
        y.append(tgt_vec.cpu().numpy()[:,:tgt_vocab.size()])

        if decode_type == 'beam_search':
            seq, score = rescore.logistic_rescore(ret_dict['topk_sequence'], log_output)
            y_vec = E.idx2vec(seq, tgt_vocab.size(), tgt_vocab.size() +1 , True)
            y_rescore.append(y_vec)
    logging("Decode type: {} , Logistic joint Decoding: {}\n".format(decode_type, logistic_joint_decoding))
    logging("{} RNN loss : {:.5f}  \nLog loss :{:.5f}\n".format(eval_type, total_rnn_loss / total, total_log_loss / total))
    
    if eval_type == 'valid' and logistic_joint_decoding is False:
        logging_valid_loss([epoch,updates,total_log_loss / total, total_rnn_loss / total])
    
    y_np = np.vstack(y)
    y_logistic_np = np.vstack(y_logistic)
    y_rnn_np = np.vstack(y_rnn)
    
    E.set_thres(0.5)
    
    def get_score(y_np, y_score_np, typ):
        logging("-"*20 + typ + '-'*20 + '\n')
        loss_dict = E.compute(y_np, y_score_np)
        logging(E.logging(loss_dict))
        return loss_dict
    
    score_rnn = get_score(y_np, y_rnn_np, 'RNN')
    get_score(y_np, y_logistic_np, 'Logistic')
    
    ## threshold 
    if eval_type == 'valid': 
        global threshold
        _,threshold = E.find_best_thres(y_np, y_logistic_np) 
    E.set_thres(threshold)
    score_logistic = get_score(y_np, y_logistic_np, 'Logistic')
    
    score_rescore = None
    if decode_type == 'beam_search':
        y_rescore_np = np.vstack(y_rescore)
        score_rescore = get_score(y_np, y_rescore_np, 'Logistic Rescore')
    
    logging('-'*50+'\n')
    
    return score_rnn, score_logistic
def test(load_checkpoint = False):
    if load_checkpoint:
        checkpoints = torch.load(log_path+'best_{}_checkpoint.pt'.format(standard_metric))
        model.load_state_dict(checkpoints['model'])
        threshold = checkpoints['threshold']
        logging("Best model was selected at {} updates.\n".format(checkpoints['updates']))
    #loss_dict = eval(0, 'test', 'greedy', config.logistic_joint_decoding)
    #loss_dict_bs = eval(0, 'test', 'beam_search', config.logistic_joint_decoding)
    loss_dict_f,_ = eval(0, 'test', 'greedy', False)
    loss_dict_bs_f,_ = eval(0, 'test', 'beam_search', False)
    loss_dict_t,_ = eval(0, 'test', 'greedy', True)
    loss_dict_bs_t,_ = eval(0, 'test', 'beam_search', True)
    #for metric in ['in_train_macro_f1', 'in_train_micro_f1','in_train_example_f1','in_train_subset_acc','in_train_hamming_loss']
    #    print()
    return loss_dict_t

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
