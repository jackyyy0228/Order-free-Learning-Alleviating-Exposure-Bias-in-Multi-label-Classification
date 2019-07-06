import sys,os

train_label_path = sys.argv[1] 
valid_label_path = sys.argv[2] 
test_label_path = sys.argv[3]

def read_label_file(path):
    L = []
    with open(path,'r') as f:
        for line in f:
            temp = []
            for t in line.rstrip().split():
                temp.append(t)
            L.append(temp)
    return L
def write_label_file(L,path):
    with open(path,'w') as f:
        for label_set in L:
            for l in label_set:
                f.write(l + ' ')
            f.write('\n')

def count_freq(L):
    d = {}
    for label_set in L:
        for l in label_set:
            if l in d:
                d[l] += 1
            else:
                d[l] = 1
    return d
def sort(L,d):
    new = []
    for label_set in L:
        l = [(x,d[x]) for x in label_set]
        l = sorted(l,key=lambda x: x[1]*-1)
        new.append([x[0] for x in l])
    return new


train_labels = read_label_file(train_label_path)
valid_labels = read_label_file(valid_label_path)
test_labels = read_label_file(test_label_path)
d = count_freq(train_labels)

train_new = sort(train_labels,d)
valid_new = sort(valid_labels,d)
test_new = sort(test_labels,d)

write_label_file(train_new,train_label_path)
write_label_file(valid_new,valid_label_path)
write_label_file(test_new,test_label_path)
