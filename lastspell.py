from __future__ import print_function

# import
import tensorflow as tf
import numpy as np
import random
from sklearn.metrics import average_precision_score, precision_recall_curve 

# custom import
import word_sort


last5, syll_check, count_word = word_sort.make_tf_data()
# last5 : ["elba$$$", "tuoba$$", ...]
# syll_check : [1, 1, ...]
# count_word : 1038

#syll_check = [random.randint(0, 1) for _ in range(count_word)]
#print(syll_check)

eng = ['$', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
dic_eng = {n: i for i, n in enumerate(eng)}
# dic_eng : {'$' : 0, 'a': 1, 'b': 2, 'c': 3, ..., 'j': 10, 'k', 11, ...}

origin_input = []
origin_target = syll_check
# origin_target : [1, 1, ...]

for t in last5:
    input = [dic_eng[i] for i in t]
    origin_input.append(np.eye(27)[input])
    
    
# Seperate test data
batch_size = int(count_word * 0.05)  # 51
randstart = random.randint(0, count_word-batch_size)
test_input = np.array(origin_input[randstart:randstart+batch_size])
test_target = np.array(origin_target[randstart:randstart+batch_size])
train_input = np.array(origin_input[:randstart] + origin_input[randstart+batch_size:])
train_target = np.array(origin_target[:randstart] + origin_target[randstart+batch_size:])

next = - batch_size
def next_batch(train_input, train_target):
    global next
    next += batch_size
    if next > count_word:
        next -= count_word
    return train_input[next:next+batch_size], train_target[next:next+batch_size]
    
    
# Set options
learning_rate = 0.01  # ?
n_hidden = 128  # hidden layer's depth? 
total_epoch = 100
n_step = len(last5[0])  # input length
# len(last5[0]) : 7

n_input = 27  # Alphabet = 26
n_class = 2  # True or False


# Modeling RNN
X = tf.placeholder(tf.float32, [None, n_step, n_input])  # X : ?
Y = tf.placeholder(tf.int32, [None])  # Y : ?

W = tf.Variable(tf.random_normal([n_hidden, n_class]))  # W : weight
b = tf.Variable(tf.random_normal([n_class]))  # B : bias


# RNN cell
cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

outputs, _ = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
# outputs: [?, 7, 128]

outputs = tf.transpose(outputs, [1, 0, 2])  # [7, ?, 128]
outputs = outputs[-1]  # [?, 128]

# model = tf.nn.tanh(tf.matmul(outputs, W) + b)
model = tf.matmul(outputs, W) + b

# print("model: " + str(model))  # [?, 2]

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

prediction = tf.cast(tf.argmax(model, 1), tf.int32)
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1, total_epoch+1):
        input_batch, target_batch = next_batch(train_input, train_target)
        _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
        predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X: input_batch, Y: target_batch})
        if epoch % 10 == 0:
            print('Epoch: {:03d} // loss: {:.6f} // training accuracy: {:.3f}'.format(epoch, loss, accuracy_val))
#             print(predict)

    predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X: test_input, Y: test_target})       
    print("테스트 정확도: %.3f%%\n"%(accuracy_val*100))
