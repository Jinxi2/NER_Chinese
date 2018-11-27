import os
import pickle
import numpy as np


# tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }


def read_corpus(path):
    data = []
    with open(path, encoding='utf-8') as f:
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
#
#
# def vocab_build(vocab_path, corpus_path, min_count):
#     data = read_corpus(corpus_path)
#     word2id = {}
#

def sentence2id(sent, word2id):
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, padmark=0):
    '''
    填充0
    :param sequences:
    :param padmark:
    :return:
    '''
    max_len = max(map(lambda x:len(x),sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [padmark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def read_dictionary(vocab_path):
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as f:
        word2id = pickle.load(f)
    print('vocab_size:', len(word2id))
    return word2id


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    seqs, labels = [], []
    for sent, tag in data:
        sent = sentence2id(sent, vocab)
        # 查询标签
        label = [tag2label[t] for t in tag]

        if len(seqs) == batch_size:
            # 返回迭代器
            yield seqs, labels
            seqs, labels = [], []
        seqs.append(sent)
        labels.append(label)
    if len(seqs) != 0:
        yield seqs, labels
