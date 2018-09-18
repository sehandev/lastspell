# import
import tensorflow as tf
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# custom import
import word_sort
import check_pr

train_data, test_data = word_sort.lastspell_data()
 
# Set options
batch_size = int(len(train_data[0]) * 0.05)
learning_rate = 0.01  # ?
n_hidden = 128  # hidden layer's depth? 
total_epoch = 1000
n_step = 5  # input length
# len(last5[0]) : 5

n_input = 27  # Alphabet = 26
n_class = 2  # True or False

next = -batch_size
def next_batch(train_input, train_target):
    global next
    next += batch_size
    check = next - len(train_data[0])
    if check > 0:
        next = -check
    return train_input[next:next+batch_size], train_target[next:next+batch_size]

   
# Modeling RNN
X = tf.placeholder(tf.float32, [None, n_step, n_input])  # X : ?
Y = tf.placeholder(tf.int32, [None])  # Y : ?

W = tf.Variable(tf.random_normal([n_hidden, n_class]))  # W : weight
b = tf.Variable(tf.random_normal([n_class]))  # B : bias


# RNN cell
cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell3 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2, cell3])

outputs, _ = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)  # outputs: [?, 5, 128]
outputs = tf.transpose(outputs, [1, 0, 2])  # [5, ?, 128]
outputs = outputs[-1]  # [?, 128]

model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

prediction = tf.cast(tf.argmax(model, 1), tf.int32)
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

array_precision, array_recall = [], []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(1, total_epoch+1):
        input_batch, target_batch = next_batch(train_data[0], train_data[1])
        _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
        predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X: input_batch, Y: target_batch})
        if epoch % 10 == 0:
          print("\n==========================")
          print('Epoch: {:03d} // loss: {:.6f} // training accuracy: {:.3f}'.format(epoch, loss, accuracy_val))
          predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X: test_data[0], Y: test_data[1]})
          print("테스트 정확도: %.3f%%"%(accuracy_val*100))

#           _precision, _recall = check_pr.sehan_precision_recall(test_target, predict)
#           array_precision.append(_precision)
#           array_recall.append(_recall)



# print("\n == array_precision ==========================")
# print(array_precision)
# print("\n == array_precision ==========================")
# print(array_recall)


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

l, = plt.plot(array_recall, array_precision, color='navy', lw=2)
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


plt.savefig('test0.png')
