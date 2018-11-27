import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
import os
import sys
from eval import conlleval
from data import batch_yield
from utils import get_logger
from data import pad_sequences


class BiLSTM_CRF():
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.hidden_dim = args.hidden_dim
        self.embeddings = embeddings
        self.CRF = args.CRF
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout
        # self.optimizer = args.optimizer
        self.lr = args.lr
        self.clip_grad = args.clip
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.vocab = vocab
        self.shuffle = args.shuffle
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config

    def build_graph(self):
        '''
        建立模型基本结构
        :return:
        '''
        self.add_placeholders()
        # 建立placeholder
        self.lookup_layer_op()
        self.BiLSTM_layer_op()
        # 建立双向LSTM结构
        self.softmax_pred_op()
        # softmax
        self.loss_op()
        # 计算loss
        self.trainstep_op()
        # one training step
        self.init_op()


    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32,shape=[None,None],name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def lookup_layer_op(self):
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout_pl)

    def BiLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            # 设置双向网络
            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=self.word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32
            )
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)

        with tf.variable_scope("proj"): # ???
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable(name='b',
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)  # 获取输出的维度
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b  # f(x) = Wx+b

            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])


    def loss_op(self):
        # crf的负对数似然
        log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                        tag_indices=self.labels,
                                                                        sequence_lengths=self.sequence_lengths)
        self.loss = -tf.reduce_mean(log_likelihood)
        # 输出误差
        tf.summary.scalar("loss",self.loss)

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)

    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optm = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            # 计算损失函数的梯度
            grads_and_vars = optm.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optm.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()



    def train(self, train, dev):
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session(config=self.config) as sess:
            # 初始化操作
            sess.run(self.init_op)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def test(self, test):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info("============= testing =============")
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        word_ids, seq_len_list = pad_sequences(seqs)
        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, padmark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)


    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        # 当前batch数

        # start_time =
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
        for step, (seqs, labels) in enumerate(batches):

            sys.stdout.write('processing: {} batches / {} batches\n'.format(step+1, num_batches))
            step_num = epoch & num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                        feed_dict=feed_dict)
            if step == 0 or step % 10 == 0 or step == num_batches:
                self.logger.info('epoch {}, step {}, loss: {:.4}, global_step: {}'.format(epoch + 1, step + 1,
                                                                                loss_train, step_num))
            # self.file_writter.add_summary(summary, step_num)

            if step == num_batches-1:
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('============= validation / test ===============')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)


    def dev_one_epoch(self, sess, dev):
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        logits, transition_params = sess.run([self.logits, self.transition_params], feed_dict=feed_dict)
        label_list = []
        for logit, seq_len in zip(logits, seq_len_list):
            viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
            label_list.append(viterbi_seq)
        return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """
        评估模型
        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)

        for _ in conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)
