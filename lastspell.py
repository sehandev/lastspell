# import
import tensorflow as tf
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# custom import
import word_sort
import check_pr

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
batch_size = int(count_word * 0.05)
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
# len(last5[0]) : 5

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
# outputs: [?, 5, 128]

outputs = tf.transpose(outputs, [1, 0, 2])  # [5, ?, 128]
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

    array_precision, array_recall = [], []
    for epoch in range(1, total_epoch+1):
        input_batch, target_batch = next_batch(train_input, train_target)
        _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
        predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X: input_batch, Y: target_batch})
        if epoch % 10 == 0:
            print('Epoch: {:03d} // loss: {:.6f} // training accuracy: {:.3f}'.format(epoch, loss, accuracy_val))
#             print(predict)
            predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X: test_input, Y: test_target})       
            print("테스트 정확도: %.3f%%\n"%(accuracy_val*100))
    
    
# precision-recall curve
            a = np.array(test_target)
            b = np.array(predict)

            print("\n\n y_test ==========================")
            print(a)
            print("\n\n y_score ==========================")
            print(b)

            average_precision = average_precision_score(a, b)
            precision, recall, _ = precision_recall_curve(a, b)

            _precision, _recall = check_pr.sehan_precision_recall(test_target, predict)
            array_precision.append(_precision)
            array_recall.append(_recall)

# plt.step(array_recall, array_precision, color='b', alpha=0.2,
#          where='post')
# plt.fill_between(array_recall, array_precision, step='post', alpha=0.2,
#                  color='b')

# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
#           average_precision))

from itertools import cycle
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

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
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


plt.savefig('test0.png')

print("size : " + str(batch_size))
