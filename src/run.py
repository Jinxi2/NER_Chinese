'''
Chinese NER
model: BiLSTM + CRF
label:
{"0", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"}
reference: https://github.com/Determined22/zh-NER-TF
time: 2018-11-27
'''
import tensorflow as tf
import os
import sys
import argparse
import time
from model import BiLSTM_CRF
from utils import str2bool, get_logger
from data import tag2label, read_corpus, read_dictionary, random_embedding

# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)

# config
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # gpu按需增长
config.gpu_options.per_process_gpu_memory_fraction = 0.2

# hyperparameters
parser = argparse.ArgumentParser(description="BiLSTM for Chinese NER task")
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
# parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='test', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
args = parser.parse_args()

word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)

## paths setting
paths = {}
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', args.train_data+"_save", timestamp)

if not os.path.exists(output_path):
    os.makedirs(output_path)
summary_path = os.path.join(output_path, 'summaries')
paths['summary_path'] = summary_path

if not os.path.exists(summary_path):
    os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")

if not os.path.exists(model_path):
    os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path

if not os.path.exists(result_path):
    os.makedirs(result_path)
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))


train_path = os.path.join('.', args.train_data, 'train_data')
test_path = os.path.join('.', args.test_data, 'test_data')
train_data = read_corpus(train_path)
test_data = read_corpus(test_path)
test_size = len(test_data)


if args.mode == 'train':
    # train
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print('train_data: {}'.format(len(train_data)))
    model.train(train=train_data, dev=test_data)
    # test_data is also dev_data, prevent overfitting
elif args.mode == 'test':
    # 加载模型
    ckpt_file = tf.train.latest_checkpoint(model_path)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print('test data: %d' % test_size)
    model.test(test_data)

