import tensorflow as tf
import numpy as np
import random

import word_sort


last5, syll_check, count_word = word_sort.make_tf_data()
# last5 : ["elba$$$", "tuoba$$", ...]
# syll_check : [1, 1, ...]

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
n_test = int(count_word * 0.05)
def seperate_test(origin_input, origin_target):
    randstart = random.randint(0, count_word-n_test)
    print(randstart)
    test_input = np.array(origin_input[randstart:randstart+n_test])
    test_target = np.array(origin_target[randstart:randstart+n_test])
    input_batch = np.array(origin_input[:randstart] + origin_input[randstart+n_test:])
    target_batch = np.array(origin_target[:randstart] + origin_target[randstart+n_test:])
    
    return input_batch, target_batch, test_input, test_target
    
# Set options
learning_rate = 0.01  # ?
n_hidden = 128  # hidden layer's depth? 
total_epoch = 500
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

outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)


# cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
# outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
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

    input_batch, target_batch, test_input, test_target = seperate_test(origin_input, origin_target)
    for epoch in range(total_epoch):
        _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
        predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X: test_input, Y: test_target})
        if epoch%10 == 0:
            print('Epoch: {:03d} // cost: {:.6f} // training accuracy: {:.3f}'.format(epoch, loss, accuracy_val))
            print('Predict : {}'.format(predict))
            print('Actual  : {}'.format(np.array(test_target)))

#         prediction = tf.cast(tf.argmax(model, 1), tf.int32)
#         prediction_check = tf.equal(prediction, Y)
#         accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

#         predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X: test_input, Y: test_target})
#         acc += accuracy_val
        
    print('%02d 정확도: %.3f%%\n\n\n'%(i+1, (accuracy_val*100)))
    