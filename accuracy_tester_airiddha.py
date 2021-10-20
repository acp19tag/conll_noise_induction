#!/usr/bin/python2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.cm as cm
from matplotlib import pyplot as plt 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense, Conv1D
from tensorflow.python.keras.layers.recurrent import GRU
from tensorflow.python.keras.layers.wrappers import TimeDistributed
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.optimizers import TFOptimizer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from conlleval import conlleval
from collections import Counter

from scripts.utils import *

# packages for learning from crowds
from crowd_layer.crowd_layers import CrowdsClassification, MaskedMultiSequenceCrossEntropy
from crowd_layer.crowd_aggregators import CrowdsCategoricalAggregator

# prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

# CONFIG PARAMETERS

NUM_RUNS = 30
DATA_PATH = "../skill_extraction_dataset_construction/dataset/"
EMBEDDING_DIM = 300
BATCH_SIZE = 64

WEIGHTS_DIR = "models/model_weights.h5"

# LOAD INDEXING WORD VECTORS

embeddings_index = {}
with open("data/glove.6B/glove.6B.%dd.txt" % (EMBEDDING_DIM,)) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print('Found %s word vectors' % len(embeddings_index))

# LOAD DATA

all_answers = read_conll(DATA_PATH+'answers.txt')
# all_mv = read_conll(DATA_PATH+'mv.txt')
# all_ground_truth = read_conll(DATA_PATH+'ground_truth.txt')
all_test = read_conll(DATA_PATH+'testset.txt')
# all_docs = all_ground_truth + all_test
all_docs = all_answers + all_test
print "Answers data size:", len(all_answers)
# print "Majority voting data size:", len(all_mv)
# print "Ground truth data size:", len(all_ground_truth)
# print "Test data size:", len(all_test)
# print "Total sequences:", len(all_docs)

# PROCESS DOCUMENTS

X_train = [[c[0] for c in x] for x in all_answers]
y_answers = [[c[1:] for c in y] for y in all_answers]
# y_mv = [[c[1] for c in y] for y in all_mv]
# y_ground_truth = [[c[1] for c in y] for y in all_ground_truth]
X_test = [[c[0] for c in x] for x in all_test]
y_test = [[c[1] for c in y] for y in all_test]
X_all = [[c[0] for c in x] for x in all_docs]
y_all = [[c[1] for c in y] for y in all_docs]

# N_ANNOT = len(y_answers[0][0])
N_ANNOT = find_number_of_annotators(y_answers)

lengths = [len(x) for x in all_docs]
all_text = [c for x in X_all for c in x]
words = list(set(all_text))
word2ind = {word: index for index, word in enumerate(words)}
ind2word = {index: word for index, word in enumerate(words)}
labels = list({c for x in y_all for c in x})

print(labels)

labels_without_o = list(set(labels) - {'O'})
# print "Labels:", labels
label2ind = {label: (index + 1) for index, label in enumerate(labels)}
ind2label = {(index + 1): label for index, label in enumerate(labels)}
ind2label[0] = "O" # padding index
# print 'Input sequence length range: ', max(lengths), min(lengths)

N_CLASSES = len(label2ind) + 1

max_label = max(label2ind.values()) + 1
# print "Max label:", max_label

maxlen = max(len(x) for x in X_all)
# print 'Maximum sequence length:', maxlen

# PREPARE EMBEDDING MATRIX

num_words = len(word2ind)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2ind.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# CONVERT DATA TO ONE-HOT ENCODING & PAD SEQUENCES

X_train_enc = onehot_encode_x(X_train, maxlen, word2ind)
# y_ground_truth_enc = onehot_encode_y(y_ground_truth, maxlen, label2ind)
y_answers_enc = onehot_encode_y_answers(y_answers, maxlen, label2ind)

X_test_enc = onehot_encode_x(X_test, maxlen, word2ind)
y_test_enc = onehot_encode_y(y_test, maxlen, label2ind)

# majority vote
y_mv = get_majority_label(y_answers)
y_mv_enc = onehot_encode_y(y_mv, maxlen, label2ind)

# BUILD BASE MODEL

def build_base_model():
    base_model = Sequential()
    base_model.add(Embedding(num_words,
                        300,
                        weights=[embedding_matrix],
                        input_length=maxlen,
                        trainable=True))
    base_model.add(Conv1D(512, 5, padding="same", activation="relu"))
    base_model.add(Dropout(0.5))
    base_model.add(GRU(50, return_sequences=True))
    base_model.add(TimeDistributed(Dense(N_CLASSES, activation='softmax')))
    base_model.compile(loss='categorical_crossentropy', optimizer='adam')

    return base_model

# EVAL DEF

def eval_model(model):
    """ returns accuracy, precision, recall, f1 """

    pr_test = model.predict(X_test_enc, verbose=2)
    pr_test = np.argmax(pr_test, axis=2)

    yh = y_test_enc.argmax(2)
    fyh, fpr = score(yh, pr_test)
    # print 'Testing accuracy:', accuracy_score(fyh, fpr)
    print 'Testing confusion matrix:'
    print confusion_matrix(fyh, fpr)

    preds_test = []
    for i in xrange(len(pr_test)):
        row = pr_test[i][-len(y_test[i]):]
        row[np.where(row == 0)] = 1
        preds_test.append(row)
    
    preds_test = [ list(map(lambda x: ind2label[x], y)) for y in preds_test]

    results_test = conlleval(preds_test, y_test, X_test, 'r_test.txt')

    print "Results for testset:", results_test

    print "Classification report:", classification_report(fyh, fpr, labels=labels_without_o)

    return accuracy_score(fyh, fpr), results_test['p'], results_test['r'], results_test['f1']

# INITIALISE OUTPUT CSV

headers = ['test_accuracy', 'model_precision', 'model_recall', 'model_f1']
output_dir = 'output/airiddha_baseline.csv'

model = build_base_model()

# # pre-train base model for a few iterations using the output of majority voting
model.fit(X_train_enc, y_mv_enc, batch_size=BATCH_SIZE, epochs=5, verbose=2)

# add crowds layer on top of the base model
model.add(CrowdsClassification(N_CLASSES, N_ANNOT, conn_type="MW"))

# instantiate specialized masked loss to handle missing answers
loss = MaskedMultiSequenceCrossEntropy(N_CLASSES).loss

# compile model with masked loss and train
model.compile(optimizer='adam', loss=loss)
model.fit(X_train_enc, y_answers_enc, batch_size=BATCH_SIZE, epochs=30, verbose=2)

# remove crowds layer before making predictions
model.pop() 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# SAVE MODEL
model.save_weights(WEIGHTS_DIR)

# results_test = eval_model(model)

test_accuracy, model_precision, model_recall, model_f1 = eval_model(model)

df = pd.DataFrame([[test_accuracy, model_precision, model_recall, model_f1]], columns=headers)

df.to_csv(output_dir, index=False, header=True)

# print "y_pred:\n"
# print np.argmax(model.predict(X_test_enc, verbose=2), axis=2)
# print "y true:\n"
# print y_test_enc.argmax(2)

# print(classification_report(y_pred=get_preds_test(model), y_true=y_test_enc.argmax(2), labels=labels_without_o))