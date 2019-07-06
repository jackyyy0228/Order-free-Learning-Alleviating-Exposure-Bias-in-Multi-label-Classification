import torch
import torch.utils.data as torch_data
import os
import numpy as np 
from torch.autograd import Variable

class dataset(torch_data.Dataset):
    def __init__(self, src, tgt, raw_src, raw_tgt, tgt_vocab_size, negative = False):

        self.src = src
        self.tgt = tgt
        self.raw_src = raw_src
        self.raw_tgt = raw_tgt
        self.tgt_vocab_size = tgt_vocab_size
        self.negative = negative
        
    def sort(self):
        data = [ z for z in zip(self.src,self.tgt,self.raw_src,self.raw_tgt)]
        data.sort(key=lambda x: len(x[0]), reverse=True) 
        self.src, self.tgt, self.raw_src, self.raw_tgt = zip(*data)
    
    def __getitem__(self, index):

        return self.src[index], self.tgt[index], \
               self.raw_src[index], self.raw_tgt[index]

    def __len__(self):
        return len(self.src)
    
def load_dataset(path):
    pass

def save_dataset(dataset, path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_padding(label_set_size,  max_len = None, 
                sos_id = None, eos_id = None):
    def pad_src(src):
        src_len = [len(s) for s in src]
        src_pad = torch.zeros(len(src), max(src_len)).long()
        for i, s in enumerate(src):
            end = src_len[i]
            src_pad[i, :end] = s[:end]
        src_len = torch.tensor(src_len)
        return src_pad, src_len

    def padding(data):
        data.sort(key=lambda x: len(x[0]), reverse=True)
        src, tgt, raw_src, raw_tgt = zip(*data)
        src_pad, src_len = pad_src(src)
        
        batch_size = len(src)
        # Vec       
        tgt_vec = np.zeros((batch_size,label_set_size),dtype = np.float32)
        for i in range(batch_size):
            for j in tgt[i]:
                tgt_vec[i][j] = 1

        tgt_vec = torch.FloatTensor(tgt_vec)
        ##RNN
        labels = np.zeros([batch_size, max_len])
        for idx, label_set in enumerate(tgt):
            labels[idx][0] = sos_id
            for t, x in enumerate(label_set):
                labels[idx][t+1] = x
            for t in range(len(label_set) + 1, max_len):
                labels[idx][t] = eos_id
        tgt_rnn = torch.LongTensor(labels)
        return raw_src, Variable(src_pad.t()), Variable(src_len), raw_tgt, Variable(tgt_vec), Variable(tgt_rnn)
    return padding

def get_loader(dataset, batch_size, shuffle, num_workers, 
               max_len = None, sos_id = None, eos_id = None):
    padding = get_padding(dataset.tgt_vocab_size,  max_len,
                          sos_id, eos_id)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=padding)
    return data_loader
    

