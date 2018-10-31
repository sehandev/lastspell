import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Lastspell:
    def __init__(self):
        '''
        < Feature >
        set learning options
        get train & test data
        run model automatically

        < Return >
        no return
        '''
        # set options
        self.train_length = 7
        self.batch_size = 1000
        self.learning_rate = 1e-3  # optimizer learning rate
        self.n_hidden = 128  # hidden layer's depth 
        self.total_epoch = 2000
        self.n_step = self.train_length # word length = 5
        self.n_input = 27  # Alphabet = 26
        self.n_class = 2  # True or False
        self.index = -self.batch_size
        
        # get data
        self.train_data, self.test_data = self.lastspell_data()
        
        # make session
        self.sess = tf.Session()
        
        # make model
        self.design_model()
        self.summary_model()
        self.run_model()
        self.make_graph()
    
    def design_model(self):
        '''
        < Feature >
        initiate LSTM multi cell, optimizer, prediction, saver

        < Return >
        no return
        '''
        # RNN cell
        with tf.name_scope("X"):
            self.X = tf.placeholder(tf.float32, [None, self.n_step, self.n_input])
        with tf.name_scope("Y"):
            self.Y = tf.placeholder(tf.int32, [None])

        with tf.name_scope("W"):
            W = tf.Variable(tf.random_normal([self.n_hidden, self.n_class]))  # W : weight
        with tf.name_scope("b"):
            b = tf.Variable(tf.random_normal([self.n_class]))  # b : bias

        cell1 = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
        cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
        cell2 = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
        cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=0.5)
        cell3 = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
        cell3 = tf.nn.rnn_cell.DropoutWrapper(cell3, output_keep_prob=0.5)
        cell4 = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
        cell4 = tf.nn.rnn_cell.DropoutWrapper(cell4, output_keep_prob=0.5)
        cell5 = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
        cell5 = tf.nn.rnn_cell.DropoutWrapper(cell5, output_keep_prob=0.5)
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2, cell3, cell4, cell5])

        with tf.name_scope("outputs"):
            outputs, _ = tf.nn.dynamic_rnn(multi_cell, self.X, dtype=tf.float32)  # outputs: [?, 5, 128]
            outputs = tf.transpose(outputs, [1, 0, 2])  # [5, ?, 128]
            outputs = outputs[-1]  # [?, 128]

        with tf.name_scope("model"):
            self.model = tf.matmul(outputs, W) + b

        # cost
        with tf.name_scope("optimizer"):
            self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model, labels=self.Y))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        # accuracy
        with tf.name_scope("accuracy"):
            self.prediction = tf.cast(tf.argmax(self.model, 1), tf.int32)
            self.prediction_check = tf.equal(self.prediction, self.Y)
            self.accuracy = tf.reduce_mean(tf.cast(self.prediction_check, tf.float32))
        
        # result
        self.array_accuracy = []
        self.array_precision = []
        self.array_recall = []

        # saver
        SAVER_DIR = "model"
        self.saver = tf.train.Saver()
        self.checkpoint_path = os.path.join(SAVER_DIR, "model")
        self.ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

    def summary_model(self):
        # make tensorboard
        self.writer = tf.summary.FileWriter('./board/lastspell', self.sess.graph)
        tf.summary.scalar('Accuracy', self.accuracy)
        tf.summary.scalar('Loss', self.cost)

        self.merged = tf.summary.merge_all()

    def run_model(self):
        '''
        < Feature >
        load prelearned model & epoch
        run model

        < Return >
        no return
        '''
        self.sess.run(tf.global_variables_initializer())

        # load model
        if self.ckpt and self.ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)

        # load epoch
        with open("./start_epoch", 'r') as f:
            start_epoch = int(f.read())

        # run model
        for epoch in range(start_epoch, self.total_epoch+1):
            # forward data to next batch
            self.index, input_batch, target_batch = self.next_batch(self.index, self.train_data[0], self.train_data[1])
            _, loss = self.sess.run([self.optimizer, self.cost], feed_dict={self.X: input_batch, self.Y: target_batch})
            
            # test with train & test data
            summary, train_accuracy = self.sess.run([self.merged, self.accuracy], feed_dict={self.X: input_batch, self.Y: target_batch})
            predict, test_accuracy = self.sess.run([self.prediction, self.accuracy], feed_dict={self.X: self.test_data[0], self.Y: self.test_data[1]})
            self.writer.add_summary(summary, epoch)

            # calculate precision and recall
            precision, recall = self.calculate_precision_recall(self.test_data[1], predict)
            self.array_precision.append(precision)
            self.array_recall.append(recall)

            if epoch % 100 == 0:
                # print information
                print("==========================")
                print("Epoch: {:03d} // loss: {:.6f}".format(epoch, loss))
                print("Training accuracy: {:.3f} // Test accuracy {:.3f}".format(train_accuracy, test_accuracy))
                print("precision : {:.3f} // recall {:.3f}".format(precision, recall))
                if epoch > 1000:
                    self.check_predict_fails(input_batch, target_batch)

            if epoch % 100000 == 0:
                # save model & epoch
                self.saver.save(self.sess, self.checkpoint_path, global_step=epoch)
                with open("./start_epoch", 'w') as f:
                    f.write(str(epoch))

    def make_graph(self):
        '''
        < Feature >
        make precision recall graph
        save graph to png file

        < Return >
        no return
        '''
        plt.figure(figsize=(7, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        lines.append(l)
        labels.append('iso-f1 curves')

        l, = plt.plot(self.array_recall, self.array_precision, color='navy', lw=2)
        lines.append(l)
        labels.append('Precision-recall for class')

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to a class')
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
        plt.savefig('result/result.png')

    def user_test(self):
        '''
        < Feature >
        test model with user's input word

        < Return >
        no return
        '''
        while True:
            print("\nWrite word to test. (Quit : q)", end=' > ')
            user_word = input()
            if user_word == 'q':
                break

            # match user's input to model
            t = self.processng_data([user_word])
            user_arr = self.data_to_eye(t)
            
            predict = self.sess.run(self.prediction, feed_dict={self.X: user_arr})

            # print test result
            if predict == [1]:
                print("받침 있음")
            elif predict == [0]:
                print("받침 없음")

    def add_data(self):
        '''
        < Feature >
        run model with new data file

        < Return >
        no return
        '''
        with open("./training_data/data_in_sorted.txt", 'r') as f:
            word_list = f.read().split('\n')[:-1]

        # match new data to model
        w = self.processng_data(word_list)
        w = self.data_to_eye(w)
        
        predict = self.sess.run(self.prediction, feed_dict={self.X: w})

        # save result from new data
        with open("./result/result_add.txt", 'w') as f:
            for i in range(len(word_list)):
                if predict[i] == 1:
                    ox = 'o'
                elif predict[i] == 0:
                    ox = 'x'
                f.write(word_list[i] + "$" + ox + "\n")
                
    def lastspell_data(self):
        '''
        < Feature >>
        Make train, test data with words data file.
        -
        Make test data with same rate of true, false.
        Shuffle data.

        < Return >
        [train_input, train_target]
        train_input : array of train words.
        train_target : array of train answers.
        [test_input, test_target]
        test_input : array of test words.
        test_target : array of test answers.
        '''
        with open("training_data/data_out.txt", 'r') as f:
            word_list = f.read().split('\n')[:-1]
            # word_list : ["able$o", "about$o", ..., ""]

        self.full_length = len(word_list)  # 단어 개수
        
        pre_true_array = []
        pre_false_array = []

        for t in word_list:
            # t : administration$o
            ox = t.split('$')[1]
            if ox == 'o':
                pre_true_array.append(t.split('$')[0])
            elif ox == 'x': 
                pre_false_array.append(t.split('$')[0])

        # precess data / ex) "information" > "noita"
        true_array = self.processng_data(pre_true_array)
        false_array = self.processng_data(pre_false_array)

        # make test data
        test_data = []
        for i in range(int(self.full_length * 0.05)):
            test_data.append([true_array.pop(), 1])
            test_data.append([false_array.pop(), 0])

        # make train data
        train_data = []
        for i in true_array:
            train_data.append([i, 1])
        for i in false_array:
            train_data.append([i, 0])

        # shuffle datas
        random.shuffle(train_data)
        random.shuffle(test_data)

        # sperate input data and target data
        train_input = []
        train_target = []
        for i in train_data:
            train_input.append(i[0])
            train_target.append(i[1])

        test_input = []
        test_target = []
        for i in test_data:
            test_input.append(i[0])
            test_target.append(i[1])

        # change array to eye array
        train_input = self.data_to_eye(train_input)
        test_input = self.data_to_eye(test_input)

        # update full_length without test_data
        self.full_length = len(train_input)

        return [train_input, train_target], [test_input, test_target]

    def processng_data(self, arr):
        '''
        < Feature >
        Change words to datas of same length.

        < Parameter >
        arr : array of words. ex) ["able", "about", ... ]
        train_length : int data of limit length. Every words will be sliced to the length.

        < Return >
        result_arr : array of words of same length. 
        '''
        result_arr = []
        for t in arr:
            # t : "able"
            t = t[::-1]
            if len(t) > self.train_length:
                t = t[:self.train_length]
            while len(t) < self.train_length:
                t += '$'
            result_arr.append(t)
        return result_arr

    def data_to_eye(self, arr):
        '''
        < Feature >
        Change words to numpy's eye arrays.

        < Parameter >
        arr : array of words. Words are composed with small alphabet and '$'(=blank).

        < Return >
        temp : array of numpy's eye arrays.
        '''
        eng = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        dic_eng = {n: i for i, n in enumerate(eng)}
        # dic_eng : {'a' : 0, 'b' : 1, 'c' : 2, ..., 'j' : 9, 'k' : 10, ...}

        temp = []
        for t in arr:
            spell = []
            for w in t:
                if w in eng:
                    spell.append(dic_eng[w])
                else:
                    spell.append(26)
            temp.append(np.eye(27)[spell])
        return temp

    def calculate_precision_recall(self, target, predict):
        '''
        < Feature >
        "calculate_precision_recall" gets lastspell model's result.
        It calculates the result's precision and recall.

        < Parameter >
        target : array of answer. "predict" will be simillar with "target."
        predict : array of predictions. lastspell model predicts answers and saves the predictions in this.

        < Return >
        precision : float data of precision.
        recall : float data of recall.
        '''
        tp, tn, fp, fn = 0, 0, 0, 0

        for i in range(len(target)):
            if predict[i] == 1:
                if target[i] == 1:
                    tp += 1
                elif target[i] == 0:
                    fp += 1
            elif predict[i] == 0:
                if target[i] == 1:
                    fn += 1
                elif target[i] == 0:
                    tn += 1

        if (tp + fp) != 0 and (tp + fn) != 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
#            print("precision : " + str(precision))
#            print("recall : " + str(recall)
        else:
            precision = 0
            recall = 0
            print("precision and recall make some ERROR")
        return precision, recall

    def next_batch(self, index, train_input, train_target):
        '''
        < Feature >
        return this batch's input & target data

        < Parameter >
        index : int data of start inder
        train_input : array of all train input data
        train_target : array of all train target data

        < Return >
        index (or check) : int data of start index
        train_input : array of this batch's input data
        train_target : array of this batch's target data
        '''
        index += self.batch_size
        check = self.full_length - index
        if check < self.batch_size:
            return check, (train_input[index:] + train_input[:self.full_length-check]), (train_target[index:] + train_target[:self.full_length-check])
        return index, train_input[index:index+self.batch_size], train_target[index:index+self.batch_size]

    def check_predict_fails(self, input_batch, target_batch):
        prediction_check = self.sess.run(self.prediction_check, feed_dict={self.X : input_batch, self.Y : target_batch})
        eng = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '$']

        for i in range(len(prediction_check)):
            if prediction_check[i] == False:
                temp = []
                print(i)
                for w in input_batch[i]:
                    for j in range(27):
                        if w[j] == 1:
                            temp.append(eng[j])
                print(temp[::-1])
                print(target_batch[i])
                print()
    



