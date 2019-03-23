import pandas as pd
import numpy as np
import pickle
import json
import random
import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

from sklearn_crfsuite.metrics import flat_classification_report

import tensorflow as tf

random.seed(8)

def savePickle(file, filename):
    pickle_out = open(filename+ ".pickle","wb")
    pickle.dump(file, pickle_out)
    pickle_out.close()


def openPickle(file):
    pickle_in = open(file+".pickle","rb")
    out = pickle.load(pickle_in)
    pickle_in.close()
    return out


def readJson(filename):
    with open(filename, "r", encoding='utf-8') as jsonfile:
        links = json.load(jsonfile)
    return links

def writeJson(filename, list):
    with open(filename,"w", encoding='utf-8') as jsonfile:
        json.dump(list,jsonfile,indent=4, separators=(',', ': '),ensure_ascii=False)

def save_model(model, filename):
    save_load_utils.save_all_weights(model,filename)

# get sentence
add = readJson("statistics_firebase323_2.json")
sentences = []
for a in add:
    if float(a["c"])>=100:
        sentence = []
        for char, tag in zip(a['a_char'], a['t']):
            sentence.append((char, tag))
        sentences.append(sentence)
random.shuffle(sentences)
writeJson("sentences.json", sentences)

words = []
for a in add:
    for char in a['a_char']:
        if float(a["c"])>=100:
            if char not in words:
                words.append(char)

words.append("ENDPAD")
savePickle(words, "words")
n_words = len(words)
# print(n_words)

#tag
tags = []
for a in add:
    for tag in a['t']:
        if float(a["c"])>=100:
            if tag not in tags:
                tags.append(tag)

tags.append("0")

n_tags = len(tags)

savePickle(tags, "tags")

# print(tags)

#see how long the sentences are


# plt.style.use("ggplot")

# plt.hist([len(s) for s in sentences], bins=50)
# plt.show()

#need equal lengths of input

max_len = 55
word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

#x 
X = [[word2idx[w[0]] for w in s] for s in sentences]
# X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)

#same for y
y = [[tag2idx[w[1]] for w in s] for s in sentences]
# y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx['0'])

#to categorical (binary)
# y = [to_categorical(i, num_classes=n_tags) for i in y]
print(X[5])
print(y[5])
#split data to training and testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

sequence_length_train = [len(i) for i in x_train]
sequence_length_test = [len(i) for i in x_test]

# Parameters used.
MODEL_PATH = 'model/bi-lstm-crf.ckpt'

def pad(sentence, max_length):
    pad_len = max_length - len(sentence)
    padding = np.zeros(pad_len)
    return np.concatenate((sentence, padding))

def batch(data, labels, sequence_lengths, batch_size, input_size):
    n_batch = int(math.ceil(len(data) / batch_size))
    index = 0
    for _ in range(n_batch):
        batch_sequence_lengths = np.array(sequence_lengths[index: index + batch_size])
        batch_length = np.array(max(batch_sequence_lengths)) # max length in batch
        batch_data = np.array([pad(x, batch_length) for x in data[index: index + batch_size]]) # pad data
        batch_labels = np.array([pad(x, batch_length) for x in labels[index: index + batch_size]]) # pad labels
        index += batch_size
        
        # Reshape input data to be suitable for LSTMs.
        batch_data = batch_data.reshape(-1, batch_length, input_size)
        
        yield batch_data, batch_labels, batch_length, batch_sequence_lengths

#################################
# Bidirectional LSTM + CRF model.
learning_rate = 0.001
training_epochs = 20
input_size = 1
batch_size = 32
num_units = 128 # the number of units in the LSTM cell
number_of_classes = n_tags

input_data = tf.placeholder(tf.float32, [None, None, input_size], name="input_data") # shape = (batch, batch_seq_len, input_size)
labels = tf.placeholder(tf.int32, shape=[None, None], name="labels") # shape = (batch, sentence)
batch_sequence_length = tf.placeholder(tf.int32) # max sequence length in batch
original_sequence_lengths = tf.placeholder(tf.int32, [None])

# Scope is mandatory to use LSTMCell (https://github.com/tensorflow/tensorflow/issues/799).
with tf.name_scope("BiLSTM"):
    with tf.variable_scope('forward'):
        lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, state_is_tuple=True)
    with tf.variable_scope('backward'):
        lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units, forget_bias=1.0, state_is_tuple=True)
    (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, 
                                                                     cell_bw=lstm_bw_cell, 
                                                                     inputs=input_data,
                                                                     sequence_length=original_sequence_lengths, 
                                                                     dtype=tf.float32,
                                                                     scope="BiLSTM")

# As we have a Bi-LSTM, we have two outputs which are not connected, so we need to merge them.
outputs = tf.concat([output_fw, output_bw], axis=2)

# Fully connected layer.
W = tf.get_variable(name="W", shape=[2 * num_units, number_of_classes],
                dtype=tf.float32)

b = tf.get_variable(name="b", shape=[number_of_classes], dtype=tf.float32,
                initializer=tf.zeros_initializer())

outputs_flat = tf.reshape(outputs, [-1, 2 * num_units])
pred = tf.matmul(outputs_flat, W) + b
scores = tf.reshape(pred, [-1, batch_sequence_length, number_of_classes])

# Linear-CRF.
log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(scores, labels, original_sequence_lengths)

loss = tf.reduce_mean(-log_likelihood)

# Compute the viterbi sequence and score (used for prediction and test time).
viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(scores, transition_params, original_sequence_lengths)

# Training ops.
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
##########################


################################
# Testing the model.
with tf.Session() as session:
    # Restore variables from disk.
    saver.restore(session, MODEL_PATH)
    
    for batch_data, batch_labels, batch_seq_len, batch_sequence_lengths in batch(x_test, y_test, sequence_length_test, len(x_test), input_size):
        tf_viterbi_sequence = session.run(viterbi_sequence, feed_dict={input_data: batch_data, 
                                                                       labels: batch_labels, 
                                                                       batch_sequence_length: batch_seq_len,
                                                                       original_sequence_lengths: batch_sequence_lengths })
    # mask to correct input sizes
    mask = (np.expand_dims(np.arange(batch_seq_len), axis=0) <
        np.expand_dims(batch_sequence_lengths, axis=1))
    total_labels = np.sum(batch_sequence_lengths)
    correct_labels = np.sum((batch_labels == tf_viterbi_sequence) * mask)
    accuracy = 100.0 * correct_labels / float(total_labels)



    print("Test accuracy: %.2f%%" % accuracy)

    print("Label:", batch_labels[0].astype(int))    
    print("Pred.:", tf_viterbi_sequence[0])

    new_x_test = []
    true = []
    pred = []
    for i in range(len(x_test)):
        w = [words[index] for index in x_test[i]]
        t_number = batch_labels[i].astype(int)[:len(x_test[i])]
        p_number = tf_viterbi_sequence[i][:len(x_test[i])]

        t = []
        for x in t_number:
            t.append(tags[x])

        p = []
        for y in p_number:
            p.append(tags[y])

        new_x_test.append(w)
        true.append(t)  
        pred.append(p)
        

    count = 0
    for i in range(len(new_x_test)):
        if pred[i] != true[i] and '中':
        # if pred[i] != true[i] and '中' not in new_X_te[i]:
            print(new_x_test[i])
            print(pred[i])
            print(true[i])
            print()
            count += 1

    print(len(new_x_test))
    print(count)

    c = 1 - (count/len(new_x_test))
    print(c)

    # f1 = f1_score(true, pred)
    # print(f1)

    report = flat_classification_report(y_pred=pred, y_true=true, digits=5)
    print(report)