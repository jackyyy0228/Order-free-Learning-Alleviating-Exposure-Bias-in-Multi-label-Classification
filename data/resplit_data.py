import numpy as np
import os,sys

def get_label():
    L = []
    with open('data/AAPD/label_all','r') as f:
        for line in f:
            tokens =  line.rstrip().split()
            L.append(tokens)
    return L
def get_out_train_idxes(label_list):
    d = {}
    for tokens in label_list:
        t = tuple(tokens)
        if t in d:
            d[t] += 1
        else:
            d[t] = 1
    L = [(k,v) for k,v in d.items()]
    L = sorted(L, key=lambda x: -x[1])

    size = 0
    out_S = set()
    while size < 4000:
        idx = np.random.randint(len(L),size=1)[0]
        n = L[idx][1]
        if size + n > 4000 or L[idx][0] in out_S:
            continue
        else:
            size += n
            out_S.add(L[idx][0])         
    out_idxes = []
    for idx, tokens in enumerate(label_list):
        t = tuple(tokens)
        if t in out_S:
            out_idxes.append(idx)
    size = 0
    in_S = set()
    in_idxes = []
    while len(in_idxes) < 4000:
        idx = np.random.randint(len(label_list),size=1)[0]
        key = tuple(label_list[idx])
        if key in out_S or idx in in_S:
            continue
        if d[key] < 3:
            continue
        d[key] -= 1
        in_idxes.append(idx)

    return in_idxes, out_idxes

def split(in_idxes,out_idxes,total_len):
    in_idxes = np.array(in_idxes)
    out_idxes = np.array(out_idxes)
    print(len(in_idxes))
    np.random.shuffle(in_idxes)
    print(len(in_idxes))
    np.random.shuffle(out_idxes)
    mid = len(in_idxes) // 2
    print(mid)   
    test_idxes = list(in_idxes[:mid])+list(out_idxes[:mid])
    val_idxes = list(in_idxes[mid:])+list(out_idxes[mid:])
    S = set(test_idxes+val_idxes)
    train_idxes = []
    for i in range(total_len):
        if i not in S:
            train_idxes.append(i)
    return train_idxes,val_idxes,test_idxes

def split_to_dir(train_idxes,val_idxes,test_idxes,label_list):
    texts = []
    with open('data/AAPD/text_all','r') as f:
        for line in f:
            texts.append(line)
    def write_text(path,idxes):
        with open(path,'w') as f:
            for i in idxes:
                f.write(texts[i])
    def write_label(path,idxes):
        with open(path,'w') as f:
            for i in idxes:
                f.write(' '.join(label_list[i])+'\n')
    if not os.path.isdir('data/AAPD2'):
        os.makedirs('data/AAPD2')
    write_text('data/AAPD2/text_train',train_idxes)
    write_text('data/AAPD2/text_val',val_idxes)
    write_text('data/AAPD2/text_test',test_idxes)
    
    write_label('data/AAPD2/label_train',train_idxes)
    write_label('data/AAPD2/label_val',val_idxes)
    write_label('data/AAPD2/label_test',test_idxes)
    
    ## set 
    S = set(train_idxes)
    for idx in train_idxes:
        S.add(tuple(label_list[idx]))
    x = 0
    for idx in val_idxes:
        if tuple(label_list[idx]) in S:
            x += 1
    print(x)
    
    x = 0
    for idx in test_idxes:
        if tuple(label_list[idx]) in S:
            x += 1
    print(x)
if __name__ == '__main__':
    np.random.seed(531)
    label_list = get_label()
    in_idxes,out_idxes = get_out_train_idxes(label_list)
    print(len(in_idxes),len(out_idxes))
    train_idxes, val_idxes,test_idxes  = split(in_idxes,out_idxes,len(label_list))
    split_to_dir(train_idxes,val_idxes,test_idxes,label_list) 


