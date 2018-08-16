import tensorflow as tf
import numpy as np

import word_sort


last5, syll_check, count_word = word_sort.make_tf_data()
# last5 : ["elba$$$", "tuoba$$", ...]
# syll_check : [1, 1, ...]
    
eng = ['$', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
dic_eng = {n: i for i, n in enumerate(eng)}
# dic_eng : {'$' : 0, 'a': 1, 'b': 2, 'c': 3, ..., 'j': 10, 'k', 11, ...}

input_batch = []
target_batch = syll_check
# target_batch : [1, 1, ...]

for t in last5:
    input = [dic_eng[i] for i in t]
    input_batch.append(np.eye(27)[input])
    
    
# Set options
learning_rate = 0.05
n_hidden = 128
total_epoch = 1000
n_step = len(last5[0])  # 7

n_input = 27  # Alphabet = 26
n_class = 2  # True or False


# Modeling RNN
X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.int32, [None])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))


# RNN cell
cell1 = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)


# cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
# outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
# outputs: [?, 7, 128]

outputs = tf.transpose(outputs, [1, 0, 2])  # [7, ?, 128]
outputs = outputs[-1]  # [?, 128]

model = tf.matmul(outputs, W) + b
# print("model: " + str(model))  # [?, 2]

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(total_epoch):
        _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
        
        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.6f}'.format(loss))

    print('==Complete==')


    prediction = tf.cast(tf.argmax(model, 1), tf.int32)
    prediction_check = tf.equal(prediction, Y)
    accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

    predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X: input_batch, Y: target_batch})
    accuracy_val *= 100
print('\n=== 예측 결과 ===')
print('정확도: %.8f%%'%accuracy_val)