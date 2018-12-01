# coding=utf-8


import os

def read_corpus_from_bjtu(path):
    data = []
    with open('train_pd.txt', encoding='GB2312') as f:
        lines = f.readlines()
    sent, tag = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent.append(char)
            tag.append(label)
        else:
            data.append((sent,tag))
            sent, tag = [], []
    return data

def data_op():
    out = open('train_bjtu','w',encoding='utf-8')
    out_test = open('test_bjtu','w',encoding='utf-8')
    l = 0
    with open('train_pd.txt', encoding='GB18030') as f:
        for line in f:
            # print(line)
            if l>80000:
                out = out_test
            words = line.split('  ')
#            print(words)
            for item in words:
                if item == '\n':continue
                word = item.split('/')
                if word[1] in ('ns','nr','nt'):
                    i = 0
                    for char in word[0]:
                        if i==0:
                            if word[1] == 'ns':
                                out.write('%s\tB-LOC\n' % char)
                            elif word[1] == 'nr':
                                out.write('%s\tB-PER\n' % char)
                            elif word[1] == 'nt':
                                out.write('%s\tB-ORG\n' % char)
                        else:
                            if word[1] == 'ns':
                                out.write('%s\tI-LOC\n' % char)
                            elif word[1] == 'nr':
                                out.write('%s\tI-PER\n' % char)
                            elif word[1] == 'nt':
                                out.write('%s\tI-ORG\n' % char)
                        i = i+1
                else:
                    for char in word[0]:
                        out.write('%s\tO\n' % char)
            l=l+1
            out.write('\n')

# path = os.path.join('.', 'data_path', 'train_data')
# read_corpus_from_bjtu(path)
data_op()
